import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from torch.utils.data import DataLoader
from torchvision import transforms
from time import time
import datetime
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import utils
import sys
from data.regular_dataset import RegularDataset
from models.models import create_model
from models.generation_model import GenerationModel
from lib.geometric_matching_multi_gpu import GMM


os.environ['CUDA_VISIBLE_DEVCIES'] = '0, 1, 2, 3'
gpu_ids = len(os.environ['CUDA_VISIBLE_DEVCIES'].split(','))


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
                            transforms.Normalize((0.5), (0.5))
    ]) # change to [C, H, W]

def train(opt):
    model = GenerationModel(opt)
    dataset = RegularDataset(opt, augment=augment)
    dataloader = DataLoader(
                        dataset,
                        shuffle=True,
                        drop_last=False,
                        num_workers=opt.num_workers,
                        batch_size=opt.batch_size_t,
                        pin_memory=True
    )
    print('the length of dataset is %d'%len(dataset))
    for epoch in range(opt.start_epoch, opt.epoch):
        print('current G learning_rate is : ', model.get_learning_rate(model.optimizer_G))
        if opt.train_mode != 'gmm':
            print('current D learning_rate is : ', model.get_learning_rate(model.optimizer_D))
        for i, data in enumerate(dataloader):
            model.set_input(opt, data)
            model.optimize_parameters(opt)
            if i % opt.print_freq == 0:
                model.print_current_errors(opt, epoch, i)
            if i % opt.val_freq == 0:
                model.save_result(opt, epoch, epoch * len(dataloader) + i)
            model.update_learning_rate(opt, model.optimizer_G, epoch)
            if opt.train_mode != 'gmm':
                model.update_learning_rate(opt, model.optimizer_D, epoch)
        if epoch % opt.save_epoch_freq == 0:
            model.save_model(opt, epoch)
                
if __name__ == '__main__':
    opt = Config().parse()
    train(opt)


