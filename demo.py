import torch
import torch.nn as nn
from models.networks import Define_G, Define_D
import torch.optim as optim
from config import Config
import os
import os.path as osp
from torch.utils.data import DataLoader
from torchvision import transforms
from data.regular_dataset import RegularDataset
from data.demo_dataset import DemoDataset
from utils.transforms import create_part
from time import time
import datetime
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import utils
from PIL import Image
from utils import pose_utils
import torch.nn.functional as F
from utils.warp_image import warped_image
from lib.geometric_matching_multi_gpu import GMM
from torchvision import utils
from PIL import Image
import time
import cv2

r"""
Forward function for vitural try-on
Note : 
      Set opt.istest == True for arbitrary pose and given image
      Set istrain = False and opt.istest == False for validating data in the validation dataset in end2end manner
"""

def load_model(model, path):

    checkpoint = torch.load(path)
    try:
        model.load_state_dict(checkpoint)
    except:
        model.load_state_dict(checkpoint.state_dict())
    model = model.cuda()

    model.eval()
    print(20*'=')
    for param in model.parameters():
        param.requires_grad = False

def forward(opt, paths, gpu_ids, refine_path):
    cudnn.enabled = True
    cudnn.benchmark = True
    opt.output_nc = 3

    gmm = GMM(opt)
    gmm = torch.nn.DataParallel(gmm).cuda()

    # 'batch'
    generator_parsing = Define_G(opt.input_nc_G_parsing, opt.output_nc_parsing, opt.ndf, opt.netG_parsing, opt.norm, 
                            not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
    
    generator_app_cpvton = Define_G(opt.input_nc_G_app, opt.output_nc_app, opt.ndf, opt.netG_app, opt.norm, 
                            not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, with_tanh=False)
    
    generator_face = Define_G(opt.input_nc_D_face, opt.output_nc_face, opt.ndf, opt.netG_face, opt.norm, 
                            not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

    models = [gmm, generator_parsing, generator_app_cpvton, generator_face]
    for model, path in zip(models, paths):
        load_model(model, path)    
    print('==>loaded model')

    augment = {}

    if '0.4' in torch.__version__:
        augment['3'] = transforms.Compose([
                                    # transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ]) # change to [C, H, W]
        augment['1'] = augment['3']

    else:
        augment['3'] = transforms.Compose([
                                # transforms.Resize(256),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]) # change to [C, H, W]

        augment['1'] = transforms.Compose([
                                # transforms.Resize(256),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
        ]) # change to [C, H, W]
    
    
    val_dataset = DemoDataset(opt, augment=augment)
    val_dataloader = DataLoader(
                    val_dataset,
                    shuffle=False,
                    drop_last=False,
                    num_workers=opt.num_workers,
                    batch_size = opt.batch_size_v,
                    pin_memory=True)
    
    with torch.no_grad():
        for i, result in enumerate(val_dataloader):
            'warped cloth'
            warped_cloth = warped_image(gmm, result) 
            if opt.warp_cloth:
                warped_cloth_name = result['warped_cloth_name']
                warped_cloth_path = os.path.join('dataset', 'warped_cloth', warped_cloth_name[0])
                if not os.path.exists(os.path.split(warped_cloth_path)[0]):
                    os.makedirs(os.path.split(warped_cloth_path)[0])
                utils.save_image(warped_cloth * 0.5 + 0.5, warped_cloth_path)
                print('processing_%d'%i)
                continue 
            source_parse = result['source_parse'].float().cuda()
            target_pose_embedding = result['target_pose_embedding'].float().cuda()
            source_image = result['source_image'].float().cuda()
            cloth_parse = result['cloth_parse'].cuda()
            cloth_image = result['cloth_image'].cuda()
            target_pose_img = result['target_pose_img'].float().cuda()
            cloth_parse = result['cloth_parse'].float().cuda()
            source_parse_vis = result['source_parse_vis'].float().cuda()

            "filter add cloth infomation"
            real_s = source_parse   
            index = [x for x in list(range(20)) if x != 5 and x != 6 and x != 7]
            real_s_ = torch.index_select(real_s, 1, torch.tensor(index).cuda())
            input_parse = torch.cat((real_s_, target_pose_embedding, cloth_parse), 1).cuda()
            
            'P'
            generate_parse = generator_parsing(input_parse) # tanh
            generate_parse = F.softmax(generate_parse, dim=1)

            generate_parse_argmax = torch.argmax(generate_parse, dim=1, keepdim=True).float()
            res = []
            for index in range(20):
                res.append(generate_parse_argmax == index)
            generate_parse_argmax = torch.cat(res, dim=1).float()

            "A"
            image_without_cloth = create_part(source_image, source_parse, 'image_without_cloth', False)
            input_app = torch.cat((image_without_cloth, warped_cloth, generate_parse), 1).cuda()
            generate_img = generator_app_cpvton(input_app)
            p_rendered, m_composite = torch.split(generate_img, 3, 1) 
            p_rendered = F.tanh(p_rendered)
            m_composite = F.sigmoid(m_composite)
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
            refine_img = p_tryon

            "F"
            generate_face = create_part(refine_img, generate_parse_argmax, 'face', False)
            generate_img_without_face = refine_img - generate_face
            source_face = create_part(source_image, source_parse, 'face', False)
            input_face = torch.cat((source_face, generate_face), 1)
            fake_face = generator_face(input_face)
            fake_face = create_part(fake_face, generate_parse_argmax, 'face', False)           
            refine_img = generate_img_without_face + fake_face

            "generate parse vis"
            if opt.save_time:
                generate_parse_vis = source_parse_vis
            else:
                generate_parse_vis = torch.argmax(generate_parse, dim=1, keepdim=True).permute(0,2,3,1).contiguous()
                generate_parse_vis = pose_utils.decode_labels(generate_parse_vis)
            "save results"
            images = [source_image, cloth_image, target_pose_img, warped_cloth, source_parse_vis, generate_parse_vis, p_tryon, refine_img]
            pose_utils.save_img(images, os.path.join(refine_path, '%d.jpg')%(i))

    torch.cuda.empty_cache()

if __name__ == "__main__":
    resume_gmm = "pretrained_checkpoint/step_009000.pth"
    resume_G_parse = 'pretrained_checkpoint/parsing.tar'
    resume_G_app_cpvton = 'pretrained_checkpoint/app.tar'
    resume_G_face = 'pretrained_checkpoint/face.tar'

    paths = [resume_gmm, resume_G_parse, resume_G_app_cpvton, resume_G_face]
    opt = Config().parse()
    if not os.path.exists(opt.forward_save_path):
        os.makedirs(opt.forward_save_path)

    forward(opt, paths, 4, opt.forward_save_path)
