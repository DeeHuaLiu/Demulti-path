# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:30:56 2023

@author: 99358
"""
import pickle

import cv2
import torch
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def uint2single(img):

    return np.float32(img/255.)
def single2uint(img):

    return np.uint8((img.clip(0, 1)*255.).round())
# convert 2/3/4-dimensional torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())
# convert single (HxWxC) to 4-dimensional torch tensor
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)


def run(img1):
    n_channels = 1
    nb = 17

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = './denoise/scunet_gray_25.pth'

    #from models.network_dncnn import DnCNN as net
    from denoise.network_scunet import SCUNet as net
    model = net(in_nc=n_channels,config=[4,4,4,4,4,4,4],dim=64)
    #model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
    # model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='BR')  # use this if BN is not merged by utils_bnorm.merge_bn(model)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)



    img1=img1*255
    img1 = np.expand_dims(img1, axis=2)  # HxWx1
    img1_L = uint2single(img1)
    img1_L = single2tensor4(img1_L)
    img1_L = img1_L.to(device)
    img1_E = model(img1_L)
    img1_E = tensor2uint(img1_E)

    img1_E=img1_E/255

    return img1_E





