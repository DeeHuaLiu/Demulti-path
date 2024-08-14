import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
import sys, os
import pickle 
import matplotlib.pyplot as plt
import time
import numpy_indexed as npd

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(delete_points_index,delete_bool,bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    ##Here are the mapped space coordinate
    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

    if not delete_bool:
        return vertices, triangles

    ##Add a judgment mechanism to find and change the points that do not meet the criteria to non-zero.
    delete_vertices_index = npd.unique(delete_points_index)
    delete_triangles_index_list = [np.where(triangles == delete_vertices_index[i])[0].reshape(-1,1) for i in range(len(delete_vertices_index))]
    delete_triangles_index = np.array([]).reshape(-1,1)
    for i in range(len(delete_triangles_index_list)):
        delete_triangles_index = np.concatenate((delete_triangles_index,delete_triangles_index_list[i]),axis=0)
    delete_triangles_index = npd.unique(delete_triangles_index)
    triangles = np.delete(triangles, delete_triangles_index.astype(int),axis=0)
    vertices[delete_vertices_index,:] = 0
    return vertices, triangles

class NeuSRenderer:
    def __init__(self,
                 sdf_network,
                 deviation_network,
                 color_network,
                 base_exp_dir,
                 expID,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.base_exp_dir = base_exp_dir
        self.expID = expID

    def render_core_sonar(self,
                        dirs,
                        pts,
                        dists,
                        sdf_network,
                        deviation_network,
                        color_network,
                        n_pixels,
                        arc_n_samples,
                        ray_n_samples,
                        cos_anneal_ratio=0.0):

        dirs_reshaped  = dirs.reshape(n_pixels, arc_n_samples, ray_n_samples, 3)
        pts_reshaped = pts.reshape(n_pixels, arc_n_samples, ray_n_samples, 3)
        dists_reshaped = dists.reshape(n_pixels, arc_n_samples, ray_n_samples, 1)

        pts_mid = pts_reshaped + dirs_reshaped * dists_reshaped/2

        pts_mid = pts_mid.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts_mid)
        sdf = sdf_nn_output[:, :1]

        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts_mid).squeeze()



        sampled_color = color_network(pts_mid, gradients, dirs, feature_vector).reshape(n_pixels, arc_n_samples, ray_n_samples)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)

        inv_s = inv_s.expand(n_pixels*arc_n_samples*ray_n_samples, 1)
        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points

        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5


        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(n_pixels, arc_n_samples, ray_n_samples).clip(0.0, 1.0)

        cumuProdAllPointsOnEachRay = torch.cat([torch.ones([n_pixels, arc_n_samples, 1]), 1. - alpha + 1e-7], -1)
    
        cumuProdAllPointsOnEachRay = torch.cumprod(cumuProdAllPointsOnEachRay, -1)

        TransmittancePointsOnArc = cumuProdAllPointsOnEachRay[:, :, ray_n_samples-2]
        
        alphaPointsOnArc = alpha[:, :, ray_n_samples-1]

        weights = alphaPointsOnArc * TransmittancePointsOnArc 

        intensityPointsOnArc = sampled_color[:, :, ray_n_samples-1]

        sdf = sdf.reshape(n_pixels*arc_n_samples, ray_n_samples)

        sdf = sdf[:, ray_n_samples-1]

        return {
            'intensityPointsOnArc': intensityPointsOnArc,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients,
            's_val': 1.0 / inv_s,
            'weights': weights,
            'cdf': c.reshape(n_pixels, arc_n_samples, ray_n_samples),
            'TransmittancePointsOnArc' :TransmittancePointsOnArc

        }

    def render_sonar(self, rays_d, pts, dists, n_pixels,
                     arc_n_samples, ray_n_samples, cos_anneal_ratio=0.0):
        # Render core
        
        ret_fine = self.render_core_sonar(rays_d,
                                        pts,
                                        dists,
                                        self.sdf_network,
                                        self.deviation_network,
                                        self.color_network,
                                        n_pixels,
                                        arc_n_samples,
                                        ray_n_samples,
                                        cos_anneal_ratio=cos_anneal_ratio)

        weights = ret_fine['weights']
        gradients = ret_fine['gradients']
        sdf = ret_fine['sdf'].cpu().detach().numpy()
        TransmittancePointsOnArc = ret_fine['TransmittancePointsOnArc']
        intensityPointsOnArc = ret_fine["intensityPointsOnArc"]
        intensityPointsOnpoint = (intensityPointsOnArc * weights).cpu().detach().numpy()

        TransmittancePointsOnArc = TransmittancePointsOnArc.cpu().detach().numpy()
        intensityPointsOnArc = intensityPointsOnArc.cpu().detach().numpy()




        #s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'sdf': sdf,
            'TransmittancePointsOnArc' : TransmittancePointsOnArc
        }, {'intensityPointsOnpoint': intensityPointsOnpoint}
    

    def extract_geometry(self, delete_points_index,delete_bool,bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(delete_points_index,
                                delete_bool,
                                bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
