import pickle

import cv2 as cv
import numpy as np
import torch
from scipy.stats import norm
from load_data import *
from helpers import DBSCAN
from pyhocon import ConfigFactory
import random
import math
import matplotlib
import pandas as pd
import logging
import argparse
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from denoise.pretreatment import *
from run_denoise import *

class PCO():
    def __init__(self, conf , if_continue = False , if_images_refer = True):
        self.conf = conf
        self.if_continue = if_continue
        ##set params
        dbscan_eps = self.conf.get_float('pco.dbscan_eps')
        self.dbscan = DBSCAN(0, 0)
        self.dbscan_1 = DBSCAN(dbscan_eps, 0)
        self.alpha = self.conf.get_int('pco.alpha')
        self.num = self.conf.get_int('pco.phi_num')

        self.dataset = self.conf.get_string('conf.dataset')
        self.data = load_data(self.dataset, False)

        self.images_reference = load_data(self.dataset, if_images_refer)['images']
        self.H = self.data['images'][0].shape[0]
        self.W = self.data['images'][0].shape[1]

        self.exp_id = self.conf.get_string('conf.expID')
        self.extrapath = './experiments/{}'.format(self.exp_id)

        ##Create data storage for PCO experiment and results
        extrapath = self.extrapath
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)
        extrapath = self.extrapath + '/PCO_p_multipath_points'
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)
        extrapath = self.extrapath + '/PCO_Cf'
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)
        extrapath = self.extrapath + '/PCO_Cs'
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)

    def fit(self):
        print('Begin Step 1\n\n')
        if self.if_continue:
           self.F_S(True)
           self.F_S(False)
           points = runner.nerf_delmultipath(False,False,False)
           points = np.concatenate((points,np.ones((points.shape[0],1))),axis=1)
           self.S_S(points)
        else:
            self.F_S(True)
            points = self.F_S(False)
            self.S_S(points)

    def F_S(self,ifdbscan):
        p_all_multipath_3points = np.array([])
        p_multipath_points = []
        for i in range(len(self.data['images'])):
            value_points_index = np.where(self.data['images'][i] > 0)
            value_points = [(value_points_index[0][i], value_points_index[1][i]) for i in
                            range(len(value_points_index[0]))]  ##Obtain the coordinates of a valid point
            
            ##The first step is to find suspected multipath points, where a rough estimate is used here.
            p_multipath_points.append(self.find_p_multipath(value_points_index, value_points,ifdbscan))  ##粗略估计一张图里面可能的multipath点
            if len(p_multipath_points[i]) > 0:
                ###
                # Convert two-dimensional image coordinates to three-dimensional coordinates and save
                # Convert matrix coordinates to sector coordinates and supplement the elevation angle coordinates, then convert them to world coordinates
                # Once you have world coordinates, they are in array form np.array([x1,y1,z1,1],
                #                                                                   [x2,y2,z2,1],
                #                                                                    ...........)"
                ###
                p_multipath_3points = self.change_coordinates(np.array(p_multipath_points[i]), -self.data["vfov"] / 2,
                                                              self.data["vfov"] / 2, self.data['sensor_poses'][i])
                if len(p_all_multipath_3points) == 0:
                    p_all_multipath_3points = p_multipath_3points
                else:
                    p_all_multipath_3points = np.concatenate(
                        (p_all_multipath_3points, p_multipath_3points))  ##Obtain the three-dimensional world coordinates of all possible multipath points.
        if ifdbscan:
            ##Store the positions of pixel points with potential multipath effects in each image.
            with open(self.extrapath+'/PCO_p_multipath_points/p_multipath_points.pkl','wb') as a:
                 pickle.dump(p_multipath_points, a)
        else:
            with open(self.extrapath+'/PCO_p_multipath_points/p_multipath_3points.pkl','wb') as a:
                 pickle.dump(p_all_multipath_3points, a)

        return p_all_multipath_3points

        # Second step, start eliminating contradictions in three-dimensional points image by image.
    def S_S(self,p_all_multipath_3points):
        p_3points_count = np.zeros((len(p_all_multipath_3points), 2), int)
        print('Begin Step 2')
        for i in range(len(self.data['images'])):
            h, w, phi = self.w_to_images(p_all_multipath_3points, self.data['sensor_poses'][i])
            h = h.astype(int)
            w = w.astype(int)
            ##Start looking for contradictions in each picture and delete the contradictory points.
            images = self.data['images'][i]
            images_reference=self.images_reference[i]
            for j in range(len(h)):
                index = (h[j].astype(int), w[j].astype(int))
                if h[j] >= self.H or h[j] <=0 or w[j] >= self.W or w[j] <=0 or phi[j] > self.data['vfov'] / 2 or phi[j] < -self.data[
                    'vfov'] / 2:  ##It must be within the range of the current sonar position's illumination."
                    continue
                if len(np.where(images[0:index[0], w[j]] > 0)[0]) == 0:
                    ##Must be that there are no obstructions ahead.
                    if self.jude_zone(images, images_reference,index, 1):       
                          p_3points_count[j,0] =  p_3points_count[j,0]+1
                    else:
                          p_3points_count[j,1] = p_3points_count[j,1]+1
                else:
                     continue
            print('Judge the {}th photo'.format(i))
        p_3points_count_all = p_3points_count[:,0]+p_3points_count[:,1]
        p_3points_count_all[np.where(p_3points_count_all==0)[0]] = 1##Prevent the occurrence of a situation where the divisor is 0.
        p_3points_count_all = p_3points_count_all.reshape(-1,1)

        p_3points_count = np.round(p_3points_count/p_3points_count_all,2)

        delet_index = np.where(p_3points_count[:,0] > self.alpha)[0] 
        p_all_multipath_3points = np.delete(p_all_multipath_3points, delet_index, axis=0)  
        p_all_multipath_3points = pd.DataFrame(p_all_multipath_3points)
        p_all_multipath_3points = p_all_multipath_3points.drop_duplicates(keep='first')
        p_all_multipath_3points = p_all_multipath_3points.values  ##Processed and removed duplicate points

        print('The second step has been completed\n\n')

        ######
        with open(self.extrapath+'/PCO_Cf/Cf_POINTS.pkl','wb') as a:
            pickle.dump(p_all_multipath_3points,a)

    def find_p_multipath(self, value_points_index, value_points, ifdbscan):
        p_multipath_points = []
        for i in range(self.W):
            column_index = [j for j, x in enumerate(value_points_index[1]) if x == i]  ##Go back to the index of the same column
            same_column_points = [value_points[j] for j in column_index]  ##Obtain the coordinate values of the valid points for each column
            if ifdbscan:
                dbscan_lable, dbscan_cluster_points = self.dbscan_1.fit(
                 same_column_points)  ##DBSCAN clustering is complete, and it returns the class of each point (in order) as well as what points are in each class.
            else:
                dbscan_lable, dbscan_cluster_points = self.dbscan.fit(
                    same_column_points)
            if len(dbscan_cluster_points) > 1:
                
                p_multipath_index = [dbscan_cluster_points[j][k] for j in range(1, len(dbscan_cluster_points)) for k in
                                     range(len(dbscan_cluster_points[j]))]
                for j in range(len(p_multipath_index)):
                    p_multipath_points.append(same_column_points[p_multipath_index[j]])  ##Get the multipath point, which is stored in a list.
        return p_multipath_points

    def change_coordinates(self, px, phi_min, phi_max, pose):
        i = px[:, 0]
        j = px[:, 1]

        phi = np.linspace(phi_min, phi_max, self.num).repeat(len(i)).reshape(len(i), -1, order='F').reshape(
            len(i) * self.num, -1)
        sonar_resolution = (self.data["max_range"] - self.data["min_range"]) / self.H
        # compute radius at each pixel
        r = i * sonar_resolution + self.data["min_range"]
        # compute bearing angle at each pixel
        theta = -self.data["hfov"] / 2 + j * self.data["hfov"] / self.W

        
        Spherical_coordinates = np.concatenate(
            (np.repeat(r, self.num).reshape(-1, 1), np.repeat(theta, self.num).reshape(-1, 1), phi), axis=1)
       
        X = Spherical_coordinates[:, 0] * np.cos(Spherical_coordinates[:, 1]) * np.cos(Spherical_coordinates[:, 2])
        Y = Spherical_coordinates[:, 0] * np.sin(Spherical_coordinates[:, 1]) * np.cos(Spherical_coordinates[:, 2])
        Z = Spherical_coordinates[:, 0] * np.sin(Spherical_coordinates[:, 2])
        Cartesian_coordinates = np.concatenate(
            (X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1), np.ones_like(X).reshape(-1, 1)), axis=1)

        
        result = np.dot(pose, Cartesian_coordinates.T)
        result = result.T

        return result

    def w_to_images(self, p_all_multipath_3points, pose):
        p_sonar_multipath_points = np.dot(np.linalg.inv(pose), p_all_multipath_3points.T)  
        r = np.sqrt(
            p_sonar_multipath_points[0, :] ** 2 + p_sonar_multipath_points[1, :] ** 2 + p_sonar_multipath_points[2,
                                                                                        :] ** 2)
        theta = np.arcsin(p_sonar_multipath_points[1, :] / np.sqrt(
            p_sonar_multipath_points[0, :] ** 2 + p_sonar_multipath_points[1, :] ** 2))
        phi = np.arcsin(p_sonar_multipath_points[2, :] / np.sqrt(
            p_sonar_multipath_points[0, :] ** 2 + p_sonar_multipath_points[1, :] ** 2 + p_sonar_multipath_points[2,
                                                                                        :] ** 2))
        r_resolution = (self.data['max_range'] - self.data['min_range']) / self.H
        h = (r - np.ones_like(r) * self.data['min_range']) / r_resolution
        w = (theta + np.ones_like(theta) * self.data['hfov'] / 2) / self.data['hfov'] * self.W
        
        h = np.round(h)
        w = np.round(w)

        return h, w, phi

    def jude_zone(self, images, images_reference,index, n):  ##Check if all elements within a square matrix range are zero, return True if all are zero
        keyA= len(np.where(images[index[0]
                                   -(n + 1):index[0]
                                   + (n + 1), index[1]
                                   - (n + 1):index[1]
                                   + (n + 1)] > 0)[0]) ==0
        keyB= images_reference[index[0],index[1]] ==0
        return keyA and keyB

    def find_p_real_multipath_points(self, p_all_multipath_points, delete_2index, images_num):
        p_all_real_multipath_points = []
        for k in range(images_num):
            p_real_multipath_points = [i for i in p_all_multipath_points[k] if i not in delete_2index]
            p_all_real_multipath_points.append(p_real_multipath_points)
        return p_all_real_multipath_points


if __name__ == '__main__':
    ##Configuration de_multi-path
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default="./confs/20deg_pool.conf")
    parser.add_argument('--is_continue', default=True, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    ###
    # If there is an initially trained scene NeRF model,
    # the corresponding data exists in the checkpoints file,
    # then one can choose to first reduce the algorithm's runtime
    # by removing some points based on the signed distance function
    # (SDF) by setting ‘if_continue’ to True.
    ###

    if_continue = False

    runner = Runner(args.conf, if_continue)
    runner.set_params()
    PCO = PCO(runner.conf, if_continue)
    PCO.fit()



