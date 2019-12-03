import torch
import torch.nn as nn
from torch.nn import init
import os
import functools
from torch.optim import lr_scheduler
from .network_g import TreeResnetGenerator, UnetGenerator
from .network_d import PatchDiscriminator, PixelDiscriminator, ResnetDiscriminator
import torch.nn.functional as F
import numpy as np

def init_weights(net, init_type='normal', init_gain=0.02):
    # initial the network
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    
    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>    

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[0,1,2,3]):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net = torch.nn.DataParallel(net).cuda()
    init_weights(net, init_type, init_gain=init_gain)

    return net

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=False)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_layer == 'none':
        norm_layer = lambda x:Identity()
    else:
        raise NotImplementedError('norm layer [%s] is not implemented'%norm_type)
    return norm_layer

class Define_G(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=4, with_tanh=True):
        super(Define_G, self).__init__()
        net = None
        norm_layer = get_norm_layer(norm_type=norm)
        self.netG = netG
        if netG == 'treeresnet':
            net = TreeResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, with_tanh=with_tanh)
        elif netG == 'unet_128':
            net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        elif netG == 'unet_256':
            net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
        self.model = init_net(net, init_type, init_gain, gpu_ids)
    
    def forward(self, x):
        return self.model(x)
        

class Define_D(nn.Module):
    def __init__(self, input_nc, ndf, netD, n_layers_D=3, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=4, n_blocks=3):
        super(Define_D, self).__init__()
        net = None
        norm_layer = get_norm_layer(norm_type=norm)
        if netD == 'basic':  # default PatchGAN classifier
            net = PatchDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
        elif netD == 'n_layers':  # more options
            net = PatchDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
        elif netD == 'pixel':     # classify if each pixel is real or fake 
            net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
        elif netD == 'resnet_blocks':
            net = ResnetDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=True, n_blocks=n_blocks)  # use_sigmoid
        else:
            raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
        self.model = init_net(net, init_type, init_gain, gpu_ids)
        
    def forward(self, x):
        return self.model(x)


