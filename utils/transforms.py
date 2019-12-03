import os
import torch
import numpy as np
from torchvision import transforms
from torchvision import utils
n_classes = 20
# colour map
label_colours = [(0,0,0)
                # 0=Background
                ,(128,0,0),(255,0,0),(0,85,0),(170,0,51),(255,85,0)
                # 1=Hat,  2=Hair, 3=Glove, 4=Sunglasses, 5=UpperClothes
                ,(0,0,85),(0,119,221),(85,85,0),(0,85,85),(85,51,0)
                # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
                ,(52,86,128),(0,128,0),(0,0,255),(51,170,221),(0,255,255)
                # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
                ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
                # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe

def create_part(source_img, source_parse, part, for_L1):
    if part == 'cloth':
        filter_part = source_parse[:,5, :, :] + source_parse[:, 6, :, :] + source_parse[:, 7, :, :]  # 5,6,7,
    elif part == 'image_without_cloth':
        filter_part = 1 - (source_parse[:,5, :, :] + source_parse[:, 6, :, :] + source_parse[:, 7, :, :])
    elif part == 'face':
        filter_part = source_parse[:, 1, :, :] + source_parse[:, 2, :, :] + source_parse[:, 4, :, :] + source_parse[:, 13, :, :] # 1,2,4,13
    elif part == 'foreground':
        filter_part = torch.sum(source_parse[:,1:,:,:], dim=1)
    elif part == 'downcloth':
        filter_part = source_parse[:,9, :, :] + source_parse[:, 12, :, :] + source_parse[:, 16, :, :] + source_parse[:, 17, :, :] + source_parse[:, 18, :, :] + source_parse[:, 19, :, :] 
    elif part == 'shoe':
        filter_part = source_parse[:,14,:,:] + source_parse[:,15,:,:] + source_parse[:,18, :, :] + source_parse[:, 19, :, :]
    elif part == 'hand':
        pass
    elif part == 'post_process':
        filter_part = torch.sum(source_parse[:,1:,:,:], dim=1)
        filter_part = torch.unsqueeze(filter_part, 1).float().cuda()
        source_img = source_img.float()
        source_img = source_img * filter_part + (1 - filter_part) * 0.8
        return source_img.float().cuda()
    if for_L1:
        filter_part = torch.unsqueeze(filter_part, 1).float().cuda()
        source_img = source_img.float()
        source_img = source_img * filter_part + (1 - filter_part)  # set 1 | white for other parts
    else:
        filter_part = torch.unsqueeze(filter_part, 1).float().cuda()
        source_img = source_img.float()
        # source_img = source_img * filter_part - (1 - filter_part) # set -1 | black for other parts
        source_img = source_img * filter_part # other part normalized 0 --> 127 gray

        # utils.save_image(source_img * 0.5 + 0.5, 'img.jpg')

    return source_img.float().cuda()

# def create_hand(parsing):
    
