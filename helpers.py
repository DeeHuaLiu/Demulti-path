import itertools
import pickle

import torch
import matplotlib
import random
import math
import numpy as np
# import open3d as o3d
from scipy.spatial import KDTree
matplotlib.use('Agg')
from MLP import *
from scipy.stats import norm
from denoise import pretreatment

torch.autograd.set_detect_anomaly(True)


def update_lr(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.0000001:
            param_group['lr'] = param_group['lr'] * lr_decay
            learning_rate = param_group['lr']
            print('learning rate is updated to ', learning_rate)
    return 0


def save_model(expID, model, i):
    # save model
    model_name = './experiments/{}/model/epoch.pt'.format(expID)
    torch.save(model, model_name)
    return 0


def get_arcs(H, W, phi_min, phi_max, r_min, r_max, c2w, n_selected_px, arc_n_samples, ray_n_samples,
             hfov, px, r_increments, randomize_points, device, cube_center):
    i = px[:, 0]
    j = px[:, 1]

    # sample angle phi
    phi = torch.linspace(phi_min, phi_max, arc_n_samples).float().repeat(n_selected_px).reshape(n_selected_px, -1)

    dphi = (phi_max - phi_min) / arc_n_samples
    rnd = -dphi + torch.rand(n_selected_px, arc_n_samples) * 2 * dphi

    sonar_resolution = (r_max - r_min) / H
    if randomize_points:
        phi = torch.clip(phi + rnd, min=phi_min, max=phi_max)

    # compute radius at each pixel
    r = i * sonar_resolution + r_min
    # compute bearing angle at each pixel
    theta = -hfov / 2 + j * hfov / W

    # Need to calculate coords to figure out the ray direction
    # the following operations mimick the cartesian product between the two lists [r, theta] and phi
    # coords is of size: n_selected_px x n_arc_n_samples x 3
    coords = torch.stack((r.repeat_interleave(arc_n_samples).reshape(n_selected_px, -1),
                          theta.repeat_interleave(arc_n_samples).reshape(n_selected_px, -1),
                          phi), dim=-1)
    coords = coords.reshape(-1, 3)

    holder = torch.empty(n_selected_px, arc_n_samples * ray_n_samples, dtype=torch.long).to(device)
    bitmask = torch.zeros(ray_n_samples, dtype=torch.bool)
    bitmask[ray_n_samples - 1] = True
    bitmask = bitmask.repeat(arc_n_samples)

    for n_px in range(n_selected_px):
        holder[n_px, :] = torch.randint(0, i[n_px] - 1, (arc_n_samples * ray_n_samples,))
        holder[n_px, bitmask] = i[n_px]

    holder = holder.reshape(n_selected_px, arc_n_samples, ray_n_samples)

    holder, _ = torch.sort(holder, dim=-1)

    holder = holder.reshape(-1)

    r_samples = torch.index_select(r_increments, 0, holder).reshape(n_selected_px,
                                                                    arc_n_samples,
                                                                    ray_n_samples)

    rnd = torch.rand((n_selected_px, arc_n_samples, ray_n_samples)) * sonar_resolution

    if randomize_points:
        r_samples = r_samples + rnd

    rs = r_samples[:, :, -1]
    r_samples = r_samples.reshape(n_selected_px * arc_n_samples, ray_n_samples)

    theta_samples = coords[:, 1].repeat_interleave(ray_n_samples).reshape(-1, ray_n_samples)
    phi_samples = coords[:, 2].repeat_interleave(ray_n_samples).reshape(-1, ray_n_samples)

    # Note: r_samples is of size n_selected_px*arc_n_samples x ray_n_samples
    # so each row of r_samples contain r values for points picked from the same ray (should have the same theta and phi values)
    # theta_samples is also of size  n_selected_px*arc_n_samples x ray_n_samples
    # since all arc_n_samples x ray_n_samples  have the same value of theta, then the first n_selected_px rows have all the same value
    # Finally phi_samples is  also of size  n_selected_px*arc_n_samples x ray_n_samples
    # but not each ray has a different phi value

    # pts contain all points and is of size n_selected_px*arc_n_samples*ray_n_samples, 3
    # the first ray_n_samples rows correspond to points along the same ray
    # the first ray_n_samples*arc_n_samples row correspond to points along rays along the same arc
    pts = torch.stack((r_samples, theta_samples, phi_samples), dim=-1).reshape(-1, 3)

    dists = torch.diff(r_samples, dim=1)
    dists = torch.cat([dists, torch.Tensor([sonar_resolution]).expand(dists[..., :1].shape)], -1)

    # r_samples_mid = r_samples + dists/2

    X_r_rand = pts[:, 0] * torch.cos(pts[:, 1]) * torch.cos(pts[:, 2])
    Y_r_rand = pts[:, 0] * torch.sin(pts[:, 1]) * torch.cos(pts[:, 2])
    Z_r_rand = pts[:, 0] * torch.sin(pts[:, 2])
    pts_r_rand = torch.stack((X_r_rand, Y_r_rand, Z_r_rand, torch.ones_like(X_r_rand)))

    pts_r_rand = torch.matmul(c2w, pts_r_rand)

    pts_r_rand = torch.stack((pts_r_rand[0, :], pts_r_rand[1, :], pts_r_rand[2, :]))

    # Centering step
    pts_r_rand = pts_r_rand.T - cube_center

    # Transform to cartesian to apply pose transformation and get the direction
    # transformation as described in https://www.ri.cmu.edu/pub_files/2016/5/thuang_mastersthesis.pdf
    X = coords[:, 0] * torch.cos(coords[:, 1]) * torch.cos(coords[:, 2])
    Y = coords[:, 0] * torch.sin(coords[:, 1]) * torch.cos(coords[:, 2])
    Z = coords[:, 0] * torch.sin(coords[:, 2])

    dirs = torch.stack((X, Y, Z, torch.ones_like(X))).T
    dirs = dirs.repeat_interleave(ray_n_samples, 0)
    dirs = torch.matmul(c2w, dirs.T).T
    origin = torch.matmul(c2w, torch.tensor([0., 0., 0., 1.])).unsqueeze(dim=0)
    dirs = dirs - origin
    dirs = dirs[:, 0:3]
    dirs = torch.nn.functional.normalize(dirs, dim=1)

    return dirs, dphi, r, rs, pts_r_rand, dists


def get_3points_ray(p_w_to_image3points, ray_n_samples, device, r_increments, c2w, cube_center, hfov, vfov, H, W,
                    arc_n_samples=1):
    h = p_w_to_image3points[:, 0].int().long()
    w = p_w_to_image3points[:, 1].int().long()
    n_selected_px = len(p_w_to_image3points)

    ##Below is the process of generating three-dimensional points along the sonar line to produce ray_n_samples sampling points.
    holder = torch.empty(n_selected_px, arc_n_samples * ray_n_samples, dtype=torch.long).to(device)
    bitmask = torch.zeros(ray_n_samples, dtype=torch.bool)
    bitmask[ray_n_samples - 1] = True
    bitmask = bitmask.repeat(arc_n_samples)

    for n_px in range(n_selected_px):
        holder[n_px, :] = torch.randint(0, h[n_px] - 1, (arc_n_samples * ray_n_samples,))
        holder[n_px, bitmask] = h[n_px]
    holder = holder.reshape(n_selected_px, arc_n_samples,
                            ray_n_samples)  ##Here, it is decomposed into nxarc_n_samplesxray_n_samples form, with each outermost layer being a point. The last column in this point is the r value of this point, and the rest are random integer values less than r.
    holder, _ = torch.sort(holder, dim=-1)  
    holder = holder.reshape(-1)  
    r_samples = torch.index_select(r_increments, 0, holder).reshape(n_selected_px,
                                                                    arc_n_samples,
                                                                    ray_n_samples) 
                                                                     ##According to the value of holder, choose the r value. This step converts the holder's index value into the corresponding r value.

    rs = r_samples[:, :, -1]
    r_samples = r_samples.reshape(n_selected_px * arc_n_samples, ray_n_samples)

    theta_samples = p_w_to_image3points[:, 1].repeat_interleave(ray_n_samples).reshape(-1, ray_n_samples)
    phi_samples = p_w_to_image3points[:, 2].repeat_interleave(ray_n_samples).reshape(-1, ray_n_samples)

    theta_samples = -hfov / 2 + theta_samples * hfov / W
    ##Here, the sampling point is complete.
    pts = torch.stack((r_samples, theta_samples, phi_samples), dim=-1).reshape(-1, 3)

    dists = torch.diff(r_samples, dim=1)
    dists = torch.cat([dists, torch.Tensor([0]).expand(dists[..., :1].shape)], -1)
    ##Here, the distance between each point is calculated.

    X_r_rand = pts[:, 0] * torch.cos(pts[:, 1]) * torch.cos(pts[:, 2])
    Y_r_rand = pts[:, 0] * torch.sin(pts[:, 1]) * torch.cos(pts[:, 2])
    Z_r_rand = pts[:, 0] * torch.sin(pts[:, 2])
    pts_r_rand = torch.stack((X_r_rand, Y_r_rand, Z_r_rand, torch.ones_like(X_r_rand))).float()

    pts_r_rand = torch.matmul(c2w, pts_r_rand)

    pts_r_rand = torch.stack((pts_r_rand[0, :], pts_r_rand[1, :], pts_r_rand[2, :]))

    # Centering step
    pts_r_rand = pts_r_rand.T - cube_center

    X = p_w_to_image3points[:, 0] * torch.cos(p_w_to_image3points[:, 1]) * torch.cos(p_w_to_image3points[:, 2])
    Y = p_w_to_image3points[:, 0] * torch.sin(p_w_to_image3points[:, 1]) * torch.cos(p_w_to_image3points[:, 2])
    Z = p_w_to_image3points[:, 0] * torch.sin(p_w_to_image3points[:, 2])

    dirs = torch.stack((X, Y, Z, torch.ones_like(X))).T
    dirs = dirs.repeat_interleave(ray_n_samples, 0).float()
    dirs = torch.matmul(c2w, dirs.T).T
    origin = torch.matmul(c2w, torch.tensor([0., 0., 0., 1.])).unsqueeze(dim=0)
    dirs = dirs - origin
    dirs = dirs[:, 0:3]
    dirs = torch.nn.functional.normalize(dirs, dim=1)
    pts = pts.cpu().numpy()

    return dirs, pts_r_rand, dists, pts


def select_coordinates(coords_all, target, N_rand, select_valid_px):
    if select_valid_px:
        coords = torch.nonzero(target)
    else:
        select_inds = torch.randperm(coords_all.shape[0])[:N_rand]
        coords = coords_all[select_inds]
    return coords


#kmeans++
class KMeansPlusPlus():
    def __init__(self, data_points, K, max_iterations):
        self.data_points = data_points
        self.K = K
        self.max_iterations = max_iterations
        self.random = random.Random()

    def cluster(self):
        # Initialize centroids list
        centroids = []
        # Randomly select first centroid
        centroids.append(self.data_points[self.random.randint(0, len(self.data_points)-1)])

        # Select remaining centroids
        for i in range(1, self.K):
            # Calculate distance of each data point to nearest centroid
            distances = [self.distance_to_nearest_centroid(centroids,dp) for dp in self.data_points]
            # Convert distances to probability distribution
            sum_distances = sum(distances)
            probabilities = [d / sum_distances for d in distances]
            # Randomly select a new centroid from probability distribution
            r = self.random.random()
            cumulative_probability = 0
            for j, p in enumerate(probabilities):
                cumulative_probability += p
                if r <= cumulative_probability:
                    centroids.append(self.data_points[j])
                    break

        # Run K-Means algorithm
        kmeans = KMeans(self.data_points, centroids, self.max_iterations)
        return kmeans.kmeans_cluster()


    def distance_to_nearest_centroid(self, centroids,dp):
        min_distance = float("inf")
        for centroid in centroids:
            distance = self.euclidean_distance(centroid,dp)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def euclidean_distance(self, coordinates1,coordinates2):
        sum_squared_distance=0
        for i in range(len(coordinates1)):
            sum_squared_distance = sum_squared_distance + math.pow(coordinates1[i]-coordinates2[i],2)
        return math.sqrt(sum_squared_distance)

class KMeans():

    def __init__(self, data_points, centroids, max_iterations):
        self.data_points=data_points
        self.centroids = centroids
        self.max_iterations=max_iterations
        self.lb=[x for x in range(len(data_points))]

    def kmeans_cluster(self):
        working_tool=KMeansPlusPlus(None,None,None)
        for _ in range(self.max_iterations):
            # Assign each data point to nearest centroid
            for i,data_point in enumerate(self.data_points):
                min_distance = float("inf")
                for j,centroid in enumerate(self.centroids):
                    distance = working_tool.euclidean_distance(data_point,centroid)
                    if distance < min_distance:
                        min_distance = distance
                        self.lb[i]=j
            # Update centroid coordinates
            self.centroids=self.update_coordinates()

        return self.centroids,self.lb

    def update_coordinates(self):
        centroids=[]

        for i in range(len(set(self.lb))):
            index=[j for j,value in enumerate(self.lb) if value==i]
            centroid_x=[self.data_points[i][0] for i in index]
            centroid_y=[self.data_points[i][1] for i in index]
            centroid_x=sum(centroid_x)
            centroid_y=sum(centroid_y)
            centroids = centroids + [(centroid_x / len(index), (centroid_y) / len(index))]

        return centroids

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        X = np.array(X)  
        labels = [0] * len(X)
        C = 0
        ID_points = []
        for idx in range(0, len(X)):
            if not (labels[idx] == 0): continue
            neighbors = self.get_neighbors(X, idx)
            if len(neighbors) < self.min_samples:
                labels[idx] = -1  # Noise
            else:
                C += 1
                self.grow_cluster(X, labels, idx, neighbors, self.eps, self.min_samples, C)
        cluster_num = sorted(set(labels))
        labels_np = np.array(labels)
        ID_points = [np.where(labels_np == j) for j in cluster_num]
        ID_points = [list(ID_points[i][0]) for i in range(len(ID_points))]  
        return labels, ID_points

    def get_neighbors(self, X, idx):
        neighbors = []
        for i in range(len(X)):
            if np.linalg.norm(X[idx] - X[i]) < self.eps:
                neighbors.append(i)
        return neighbors

    def grow_cluster(self, X, labels, idx, neighbors, eps, min_samples, C):
        labels[idx] = C
        i = 0
        neighbors_same_cluster = []
        while i < len(neighbors): 
            next_point = neighbors[i]
            if labels[next_point] == -1:
                labels[next_point] = C
            elif labels[next_point] == 0:
                labels[next_point] = C
                new_neighbors = self.get_neighbors(X, next_point)
                if len(new_neighbors) >= min_samples:
                    neighbors = neighbors + new_neighbors
            i += 1
        return neighbors


def smooth(denoise_images):
    images = denoise_images
    s = images[0].shape[0]
    for index in range(len(images)):
        images[index] = pretreatment.run(images[index])  ##Here, add denoise preprocessing to denoise.
        images[index][s - 185:, :] = 0
    return images


def probabilty(d, distance): 
    sigma = distance / 3
    mu = 0
    probability = (1 - norm.cdf(d, mu, sigma)) * 2
    return probability


def getfortime(num, n):
    list_num = num // (n - 1)
    list_num_remain = num % (n - 1)
    if list_num == 0:
        list_fortimes = num * [1]
        list_fortimes = list(itertools.accumulate(list_fortimes))
        list_fortimes[0] = 0
        return list_fortimes
    else:
        list_fortimes = [list_num] * (n - 1)
    if not list_num_remain == 0:
        list_fortimes.append(list_num_remain)
    list_fortimes = list(itertools.accumulate(list_fortimes))
    if not list_fortimes[0] == 1:
        list_fortimes.insert(0, 1)
    list_fortimes[0] = 0
    return list_fortimes


def whether_tranmist_voxel(points_sonar, ray_sonar, dists, R, s, r, points_num , select_num):
    fortimes = getfortime(ray_sonar.shape[0], points_num + 1)
    X_r_rand = ray_sonar[:, 0] * np.cos(ray_sonar[:, 1]) * np.cos(ray_sonar[:, 2])
    Y_r_rand = ray_sonar[:, 0] * np.sin(ray_sonar[:, 1]) * np.cos(ray_sonar[:, 2])
    Z_r_rand = ray_sonar[:, 0] * np.sin(ray_sonar[:, 2])
    ray_sonar_Car = np.concatenate([X_r_rand.reshape(-1, 1),
                                    Y_r_rand.reshape(-1, 1),
                                    Z_r_rand.reshape(-1, 1),
                                    np.ones_like(Z_r_rand).reshape(-1, 1)], axis=1)
    Transimit_pro_all = np.array([]).reshape(-1, 1)
    for index in range(len(fortimes) - 1):
        dists_ray = dists[index, :].T.reshape(-1,1)
        ray_sonar_part = ray_sonar_Car[fortimes[index]:fortimes[index + 1], :]
        theta = ray_sonar[fortimes[index], 1]
        phi = ray_sonar[fortimes[index] , 2]
        pose_z = np.array([[math.cos(theta), math.sin(theta), 0, 0],
                           [-math.sin(theta), math.cos(theta), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        pose_y = np.array([[math.cos(phi), 0, math.sin(phi), 0],
                           [0, 1, 0, 0],
                           [-math.sin(phi), 0, math.cos(phi), 0],
                           [0, 0, 0, 1]])
        pose = np.dot(pose_y, pose_z)
        ##The calculation of coordinates with rays as the X-axis
        points_ray = np.dot(pose, points_sonar.T).T[:, 0:3]
        ray_part_ray = np.dot(pose, ray_sonar_part.T).T[:, 0:3]
        ##Define the cylindrical surface function
        infun = lambda points: points[:, 1] ** 2 + points[:, 2] ** 2
        ##Calculate whether it is inside the ellipse
        in_zone_distance = infun(points_ray)
        in_zone_index = np.where(in_zone_distance <= R ** 2)
        ##Calculate the total number inside the cluster
        in_zone_points = points_ray[in_zone_index[0], :]
        ##Calculate the coordinates after translation, and translate to each ray sampling point.
        in_zone_points = in_zone_points[:,np.newaxis,:]
        in_zone_points = in_zone_points - ray_part_ray
        in_zone_points = np.array([in_zone_points[:,i,:] for i in range(len(ray_part_ray))])
        ##Calculate the density within the cylindrical volume of each sampling point.
        # v = math.pi*R**2*s
        E = math.pi*(R)**2
        num = [np.sum((np.abs(in_zone_points[i,:,0])<=s/2)) for i in range(len(ray_part_ray))]

        A = math.pi*(r)**2
        ##Calculate Pho
        deta = np.array(num).reshape(-1,1)*A/E/s
        ##Calculate Transittance 
        Tramsit_pro = dists_ray*deta.reshape(-1,1)
        Tramsit_pro = np.exp(-Tramsit_pro)
        Tramsit_pro = np.cumprod(Tramsit_pro)[-select_num].reshape(-1,1)
        Transimit_pro_all = np.concatenate((Transimit_pro_all,Tramsit_pro),axis=0)
    return Transimit_pro_all


class DBSCAN_pp:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self.labels_ = np.full(X.shape[0], -1, dtype=int)
        self.cluster_id_ = 0

       
        tree = KDTree(X)

      
        for i in range(X.shape[0]):
            if self.labels_[i] != -1:
                continue  

            
            neighbors_idx = tree.query_ball_point(X[i], self.eps)
            

            
            self.expand_cluster(X, tree, neighbors_idx, i)

        return self

    def expand_cluster(self, X, tree, neighbors_idx, point_idx):
        
        self.labels_[point_idx] = self.cluster_id_

        
        neighbors_idx = [idx for idx in neighbors_idx if self.labels_[idx] !=-1]

        
        self.labels_[neighbors_idx] = self.cluster_id_

        
        self.cluster_id_ += 1
       

    def get_cluster_centers(self, X):
        """计算并返回每个聚类的中心（均值点）"""
        unique_labels = np.unique(self.labels_[self.labels_ != -1])  
        cluster_centers = np.zeros((unique_labels.shape[0], 3))
        cluster_points_all = []
        for index,label in enumerate(unique_labels):
            cluster_points = X[self.labels_ == label]  
            cluster_points_all.append(cluster_points)
            cluster_centers[index] = np.mean(cluster_points, axis=0)  
        return cluster_centers , cluster_points_all










