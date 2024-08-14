import math
import os, sys
import pickle

import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from tqdm import tqdm, trange
import scipy.io
import matplotlib.pyplot as plt
from helpers import *
from MLP import *
# from PIL import Image
import cv2 as cv
import time
import random
import string
import numpy_indexed as npd
from pyhocon import ConfigFactory
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
import trimesh
from itertools import groupby
from operator import itemgetter
from load_data import *
import logging
import argparse

from pco_reconstruction import *
from math import ceil
from sklearn.cluster import KMeans


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

class Runner:
    def __init__(self, conf, is_continue = False, intensity_fine=False, write_config=True):
        conf_path = conf
        f = open(conf_path)
        conf_text = f.read()
        self.is_continue = is_continue
        self.conf = ConfigFactory.parse_string(conf_text)
        self.write_config = write_config

    def set_params(self):
        self.expID = self.conf.get_string('conf.expID')

        dataset = self.conf.get_string('conf.dataset')
        self.image_setkeyname = self.conf.get_string('conf.image_setkeyname')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset

        # Training parameters
        self.sdf_distance = self.conf.get_float('transmittance.sdf_distance')
        self.beta = self.conf.get_float('transmittance.beta')
        self.R = self.conf.get_float('transmittance.R')
        self.ray_samples = self.conf.get_int('transmittance.ray_samples')
        self.lambda_1 = self.conf.get_float('transmittance.lambda_1')
        self.lambda_2 = self.conf.get_float('transmittance.lambda_2')
        self.select_num = self.conf.get_int('transmittance.select_num')

        self.end_iter = self.conf.get_int('train.end_iter')
        self.N_rand = self.conf.get_int('train.num_select_pixels')  # H*W
        self.arc_n_samples = self.conf.get_int('train.arc_n_samples')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.percent_select_true = self.conf.get_float('train.percent_select_true', default=0.5)
        self.r_div = self.conf.get_bool('train.r_div')
        self.dbscan_eps = self.conf.get_float('pco.dbscan_eps')
        self.pd_points_arc = self.conf.get_int('pco.phi_num')
        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')

        self.variation_reg_weight = self.conf.get_float('train.variation_reg_weight')
        self.px_sample_min_weight = self.conf.get_float('train.px_sample_min_weight')

        self.ray_n_samples = self.conf['model.neus_renderer']['n_samples']  
        self.base_exp_dir = './experiments/{}'.format(self.expID)
        self.randomize_points = self.conf.get_float('train.randomize_points')
        self.select_px_method = self.conf.get_string('train.select_px_method')
        self.select_valid_px = self.conf.get_bool('train.select_valid_px')
        self.x_max = self.conf.get_float('mesh.x_max')
        self.x_min = self.conf.get_float('mesh.x_min')
        self.y_max = self.conf.get_float('mesh.y_max')
        self.y_min = self.conf.get_float('mesh.y_min')
        self.z_max = self.conf.get_float('mesh.z_max')
        self.z_min = self.conf.get_float('mesh.z_min')
        self.level_set = self.conf.get_float('mesh.level_set')

        self.demultipath = PCO(self.conf , if_images_refer = False)
        self.data = self.demultipath.data
        self.H, self.W = self.data[self.image_setkeyname][0].shape

        self.r_min = self.data["min_range"]  
        self.r_max = self.data["max_range"]  
        self.phi_min = -self.data["vfov"] / 2  
        self.phi_max = self.data["vfov"] / 2  
        self.vfov = self.data["vfov"]  
        self.hfov = self.data["hfov"]

        self.cube_center = torch.Tensor(
            [(self.x_max + self.x_min) / 2, (self.y_max + self.y_min) / 2, (self.z_max + self.z_min) / 2])  ##

        self.timef = self.conf.get_bool('conf.timef')
        self.end_iter = self.conf.get_int('train.end_iter')
        self.start_iter = self.conf.get_int('train.start_iter')

        self.object_bbox_min = self.conf.get_list('mesh.object_bbox_min')
        self.object_bbox_max = self.conf.get_list('mesh.object_bbox_max')

        r_increments = []  
        self.sonar_resolution = (self.r_max - self.r_min) / self.H 
        for i in range(self.H):
            r_increments.append(i * self.sonar_resolution + self.r_min)

        self.r_increments = torch.FloatTensor(r_increments).to(self.device)  

        #The saved directory of the data after creating the experiment
        extrapath = './experiments/{}'.format(self.expID)
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)

        extrapath = './experiments/{}/checkpoints'.format(self.expID)
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)

        extrapath = './experiments/{}/model'.format(self.expID)
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)


        if self.write_config:
            with open('./experiments/{}/config.json'.format(self.expID), 'w') as f:
                json.dump(self.conf.__dict__, f,
                          indent=2)  
        # Create all image tensors beforehand to speed up process

        self.i_train = np.arange(
            len(self.data[self.image_setkeyname])) 
        self.coords_all_ls = [(x, y) for x in np.arange(self.H) for y in np.arange(self.W)]
        self.coords_all_set = set(self.coords_all_ls)

        # self.coords_all = torch.from_numpy(np.array(self.coords_all_ls)).to(self.device)

        self.del_coords = []
        for y in np.arange(self.W):
            tmp = [(x, y) for x in np.arange(0, self.ray_n_samples)]  
            self.del_coords.extend(tmp)  

        self.coords_all = list(self.coords_all_set - set(self.del_coords)) 
        self.coords_all = torch.LongTensor(self.coords_all).to(self.device)  # nx2

        self.criterion = torch.nn.L1Loss(reduction='sum')

        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)

        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)  ##误差网络
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(
            self.device)  ##redering network
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.iter_step = 0
        self.renderer = NeuSRenderer(self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.base_exp_dir,
                                     self.expID,
                                     **self.conf['model.neus_renderer'])
        latest_model_name = None
        if self.is_continue:
            model_list_raw = os.listdir(
                os.path.join(self.base_exp_dir, 'checkpoints'))  
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth':  # and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

    def getRandomImgCoordsByPercentage(self, target, image_i):  
        true_coords = []
        for y in np.arange(self.W):
            col = target[:, y]
            gt0 = col > 0
            indTrue = np.where(gt0)[0]  
            if len(indTrue) > 0:
                true_coords.extend([(x, y) for x in indTrue])  

        sampling_perc = int(self.percent_select_true * len(true_coords))
        true_coords = random.sample(true_coords, sampling_perc)
        true_coords = list(set(true_coords) - set(self.del_coords)) 
        true_coords = torch.LongTensor(true_coords).to(self.device)
        target = torch.Tensor(target).to(self.device)
        if self.iter_step % len(self.data[self.image_setkeyname]) != 0:
            N_rand = 0
        else:
            N_rand = self.N_rand
        N_rand = self.N_rand
        coords = select_coordinates(self.coords_all, target, N_rand,
                                    self.select_valid_px)  

        coords = torch.cat((coords, true_coords), dim=0) 

        coords_list = np.array(coords.cpu())
        return coords, target, coords_list


    def in_zone_points(self,p_w_to_image3points):
        delh_index_a = np.where(p_w_to_image3points[:, 0] >= self.H)[0].reshape(-1, 1)
        delh_index_b = np.where(p_w_to_image3points[:, 0] <= 0)[0].reshape(-1, 1)
        delw_index_a = np.where(p_w_to_image3points[:, 1] >= self.W)[0].reshape(-1, 1)
        delw_index_b = np.where(p_w_to_image3points[:, 1] <= 0)[0].reshape(-1, 1)
        delphi_index_a = np.where(p_w_to_image3points[:, 2] > self.data['vfov'] / 2)[0].reshape(-1, 1)
        delphi_index_b = np.where(p_w_to_image3points[:, 2] < -self.data['vfov'] / 2)[0].reshape(-1, 1)
        delindex = np.concatenate(
            (delphi_index_a, delphi_index_b, delh_index_a, delh_index_b, delw_index_a, delw_index_b), axis=0)
        delindex = npd.unique(delindex)

        return delindex

    def nerf_delmultipath(self,if_delbegin = True,if_dbscan = True,if_continue = False):
        ##Determine whether to continue running the process again after having PCO data that has been processed with SDF.
        if not if_continue:
            if if_delbegin:
              with open(self.base_exp_dir+'/PCO_Cf/Cf_POINTS.pkl', 'rb') as f:
                   p_all_multipath_3points = pickle.load(f)
            else:
              with open(self.base_exp_dir+'/PCO_p_multipath_points/p_multipath_3points.pkl', 'rb') as f:
                   p_all_multipath_3points = pickle.load(f)
            fortimes = getfortime(len(p_all_multipath_3points), 30)
            three_points = np.array([]).reshape(-1, 3)

            for index in range(len(fortimes) - 1):
                p_multipath_3points_part = p_all_multipath_3points[fortimes[index] - 1:fortimes[index + 1] - 1][:, 0:3]
                cube_center = self.cube_center.cpu().numpy().reshape(1, 3)
                p_multipath_3points_part = p_multipath_3points_part - np.repeat(cube_center, len(p_multipath_3points_part),
                                                                                axis=0)
                p_multipath_3points_part = torch.tensor(p_multipath_3points_part, dtype=torch.float32)
                sdf = self.sdf_network.sdf(p_multipath_3points_part).cpu().detach().numpy()
                sdf = sdf.reshape(-1, 1)
                del_index_a = np.where(sdf >= self.sdf_distance)[0].reshape(-1,1)
                del_index = del_index_a
                p_multipath_3points_part = p_multipath_3points_part.cpu().detach().numpy()
                p_multipath_3points_part = np.delete(p_multipath_3points_part, del_index, axis=0)
                p_multipath_3points_part = p_multipath_3points_part + np.repeat(cube_center, len(p_multipath_3points_part),
                                                                                axis=0)
                three_points = np.concatenate([three_points, p_multipath_3points_part])
            if not if_delbegin:
                return three_points

            print('Begin DBSCAN')
            dbscan = DBSCAN_pp(0.035,0)
            dbscan.fit(three_points)
            three_pints, clsuter_points = dbscan.get_cluster_centers(three_points)
            with open(self.base_exp_dir+'/PCO_Cs/Cs_POINTS.pkl', 'wb') as f:
                pickle.dump(three_points,f)
        else:
            with open(self.base_exp_dir + '/PCO_Cs/Cs_POINTS.pkl', 'rb') as f:
                three_points = pickle.load(f)
        p_all_multipath_3points = np.concatenate([three_points, np.ones(len(three_points)).reshape(-1, 1)], axis=1)
        print('The first step is done\n\nProbably multi-path points in 3D space：{}'.format(len(p_all_multipath_3points)))

    
        # Step 2: Remove points of conflict.
        print('Begin step 2')
        with open(self.base_exp_dir+'/PCO_p_multipath_points/p_multipath_points.pkl', 'rb') as f2:
            p_multipath_points = pickle.load(f2)
        from scipy.spatial import KDTree
        tree = KDTree(three_points)
        min_ditance, index = tree.query(three_points, k=2)
        s = np.sort(min_ditance[:, 1])
        s = s[-1]
        for i in range(len(self.data['images'])):

            ###
            # After processing every five sonar images, save the results into the file
            # named 'self.base_exp_dir+'{}_de_part.pkl'.format(self.expID)' which can be
            # directly read and viewed for the processing results using cv2.
            ###
            if i-1 %5 ==0:
                with open(self.base_exp_dir+'/{}_de_part.pkl'.format(self.expID), 'wb') as fi4:
                    pickle.dump(self.data, fi4)
            h, w, phi = self.demultipath.w_to_images(p_all_multipath_3points, self.data['sensor_poses'][i])
            h = h.astype(int)
            w = w.astype(int)
            p_w_to_image3points = np.concatenate((h.reshape(-1,1),w.reshape(-1,1),phi.reshape(-1,1),np.ones_like(phi.reshape(-1,1))),axis=1)
            ##Here, the points outside the current sonar range are removed
            delindex = self.in_zone_points(p_w_to_image3points)

            p_w_to_image3pointsimgaes = np.delete(p_w_to_image3points,delindex,axis=0)
            p_w_to_image3points_irang = np.delete(p_all_multipath_3points,delindex,axis=0)
            
            inspect_points = p_w_to_image3pointsimgaes[:,0:2]
            inspect_points_addphi = p_w_to_image3pointsimgaes[:,0:3]

            AP_p_multipath_points = p_multipath_points[i]
            pose = self.data['sensor_poses'][i]
            p_all_multipath_3points__sonar = np.dot(np.linalg.inv(pose), p_w_to_image3points_irang.T)

            if len(AP_p_multipath_points) ==0:
                continue
            else:
               tree2 = KDTree(inspect_points)
               AP_p_multipath_points_array =np.array(AP_p_multipath_points)
               index = tree2.query_ball_point(AP_p_multipath_points_array, 1e-5)


         
            zero_index = [k for k in range(len(index)) if len(index[k])==0]
            value_index = [k for k in range(len(index))]
            value_index = list(set(value_index) - set(zero_index))
            self.data['images'][i][AP_p_multipath_points_array[zero_index,0],
                                   AP_p_multipath_points_array[zero_index,1]] = 0

            for j in value_index:
                inspect_points_addphi_part = inspect_points_addphi[index[j], :].reshape(-1, 3)

                ##Add the judgment of certain impossible contributions of M(x) values from a single image
                dirs, pts_r_rand, dists, _ = get_3points_ray(torch.tensor(inspect_points_addphi_part).to(self.device),
                                                       64,
                                                       self.device,
                                                       self.r_increments, torch.tensor(pose),
                                                       self.cube_center, self.data['hfov'],self.data['vfov'],self.H,self.W)
                _, _, dists_images, pts_images = get_3points_ray(
                                                       torch.tensor(inspect_points_addphi_part).to(self.device),
                                                       self.ray_samples,
                                                       self.device,
                                                       self.r_increments, torch.tensor(pose),
                                                       self.cube_center, self.data['hfov'], self.data['vfov'], self.H, self.W)

                render_out,_ = self.renderer.render_sonar(dirs, pts_r_rand, dists, len(inspect_points_addphi_part),
                                                        arc_n_samples=1, ray_n_samples=self.ray_n_samples,
                                                        cos_anneal_ratio=self.get_cos_anneal_ratio())

                dists_images = dists_images.cpu().numpy()
                ## Parameters related to s, the one before is the length of the cylinder, and the one after is the radius of the particle.
                TranmistOnArc_images = whether_tranmist_voxel(p_all_multipath_3points__sonar.T,
                                                              pts_images, dists_images, self.R,
                                                              s, s/2,len(inspect_points_addphi_part),
                                                              self.select_num)
                TranmistOnArc_nerf = render_out['TransmittancePointsOnArc']
                TranmistOnArc_all = self.lambda_1*TranmistOnArc_nerf + self.lambda_2*TranmistOnArc_images
                if any(TranmistOnArc_all >= self.beta):
                    continue
                else:
                    self.data['images'][i][AP_p_multipath_points[j]] = 0
            print('The {}th image has been completed.'.format(i+1))
        with open(self.base_exp_dir+'/{}_de_all.pkl','wb') as fi:
            pickle.dump(self.data, fi)
    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def validate_mesh(self, delete_points_index, delete_bool, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(delete_points_index, delete_bool, bound_min, bound_max,
                                           resolution=resolution,
                                           threshold=threshold)
        if not delete_bool:
            return vertices, triangles
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.fill_holes()
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step + 2)))


if __name__ == '__main__':

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
    runner = Runner(args.conf, args.is_continue)
    runner.set_params()
    runner.nerf_delmultipath(if_continue = False)

