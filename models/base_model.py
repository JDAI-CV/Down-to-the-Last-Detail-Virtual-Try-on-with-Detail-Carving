import os
import os.path as osp
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils import pose_utils
from utils.loss import GANLoss, PixelWiseBCELoss, PixelSoftmaxLoss, VGGLoss, NNLoss, NewL1Loss, TVLoss
from abc import ABC, abstractmethod
import datetime
from torch.optim import lr_scheduler
import time

class BaseModel(ABC):
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        cudnn.enabled = True
        cudnn.benchmark = True
        # define loss
        self.criterionGAN = GANLoss(opt.gan_mode).cuda()
        self.criterionL1 = torch.nn.L1Loss().cuda()
        self.criterion_newL1 = NewL1Loss()
        self.criterion_smooth_L1 = torch.nn.SmoothL1Loss()
        self.criterion_vgg = VGGLoss("vgg_model/vgg19-dcbb9e9d.pth").cuda()
        self.weight = np.array([0.03] * 2 + [0.08] * 1 + [0.03] * 2 + [0.08] * 3 + [0.03] * 5 + [0.08] * 5 + [0.05] * 2)
        self.weight = torch.Tensor(self.weight).cuda()
        self.criterionBCE_re = PixelSoftmaxLoss(self.weight).cuda()
        self.criterion_tv = TVLoss()
        # log dir
        if opt.joint_all:
            self.save_dir = os.path.join('net_model', 'joint_checkpoint', opt.suffix)
        else:   
            self.save_dir = os.path.join('net_model', opt.train_mode + '_checkpoint', opt.suffix)
        self.date_suffix = self.dt()
        self.log_dir = os.path.join(self.save_dir, 'logs')

        self.log_name = os.path.join(self.log_dir, 'train_log_%s_%s.log'%(opt.suffix, self.date_suffix))
        self.vis_path = os.path.join(self.save_dir, 'imgs')
        
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        with open(os.path.join(self.log_name), 'a') as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
    
    def dt(self):
        return datetime.datetime.now().strftime("%m-%d-%H")

    def update_learning_rate(self, opt, optimizer, epoch):
        base_lr = opt.lr
        lr = base_lr
        if epoch > 30:
            lr = base_lr * (1 - base_lr/opt.decay_iters)
        if isinstance(optimizer, list):
            for _ in optimizer:
                for param_group in _.param_groups:
                    param_group['lr'] = lr
        else:   
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # TO DO
        # if epoch <= 4:
        #     lr = base_lr
        # elif epoch > 24:
        #     lr = 2e-4 * (1 - (epoch-24) / 10)
        # else:
        #     lr = self.get_learning_rate(optimizer)
        #     if lr <= 2e-4:
        #         lr = 2e-4
        #     else:
        #         lr = base_lr - (base_lr - 2e-4) * (epoch - 4) / 15

        # for scheduler in self.get_scheduler(optimizer, opt):
        #     scheduler.step()
        
    def get_learning_rate(self, optimizer):
        lr = []
        if not isinstance(optimizer, list):
            for param_group in optimizer.param_groups:
                lr += [param_group['lr']]
        else:
            for _ in optimizer:
                for param_group in _.param_groups:
                    lr += [param_group['lr']]
        return lr[0]

    def set_requires_grad(self, nets, requires_grad=True):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def adjust_fade_in_alpha(self, epoch):
        alpha = list(np.arange(0.6, 1.2, 0.2))
        if epoch < 6:
            fade_in_alpha = alpha[epoch // 2]
        else:
            fade_in_alpha = 1
        
        return int(fade_in_alpha)

    def load_network(self, network, save_path, ifprint=False):
        if ifprint:
            print(network)       
        try:
            network.load_state_dict(torch.load(save_path))
        except:
            pretrained_dict = torch.load(save_path)                
            model_dict = network.state_dict()
            try:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                network.load_state_dict(pretrained_dict)
            except:
                print('Pretrained network %s has fewer layers; The following are not initialized:')
                for k, v in pretrained_dict.items():                      
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v
                        print(k)
                if sys.version_info >= (3,0):
                    not_initialized = set()
                else:
                    from sets import Set
                    not_initialized = Set()                    

                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        not_initialized.add(k.split('.')[0])

                print(sorted(not_initialized))
                network.load_state_dict(model_dict)     
    
    def print_loss(self, opt):
        pass
    
    def get_scheduler(self, optimizer, opt):

        if opt.lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler
    
        # errors: same format as |errors| of plotCurrentErrors
    



                

