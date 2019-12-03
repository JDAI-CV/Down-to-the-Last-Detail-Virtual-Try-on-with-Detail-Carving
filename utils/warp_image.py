#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time

import numpy as np
from PIL import Image
from .pose_utils import heatmap_embedding
from torchvision import transforms
from torch.utils.data import DataLoader


def warped_image(model, result):
    with torch.no_grad():
        model.eval()
        # input
        target_pose_embedding = result['target_pose_embedding'].float().cuda()
        source_parse_shape = result['source_parse_shape'].cuda()
        cloth_image = result['cloth_image'].float().cuda()
        im_h = result['im_h'].cuda()
        
        # cloth-agnostic representation
        agnostic = torch.cat([source_parse_shape, im_h, target_pose_embedding], 1)

        grid, theta = model(agnostic, cloth_image)

        warped_cloth = F.grid_sample(cloth_image, grid, padding_mode='border')
        
    return warped_cloth






