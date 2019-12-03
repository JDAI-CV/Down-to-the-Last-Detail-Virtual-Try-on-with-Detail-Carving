import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import os.path as osp
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision import utils
from utils import pose_utils

class ClothDataset(Dataset):
    def __init__(self, istrain, augment, train_mode):
        self.transforms = augment
        # pair_list =[i.strip() for i in open('../../input/MPV/selected_pair.txt', 'r').readlines()] 
        pair_list =[i.strip() for i in open('demo/demo.txt', 'r').readlines()] 
        train_list = list(filter(lambda p: p.split('\t')[3] == 'train', pair_list))
        train_list = [i for i in train_list]
        test_list = list(filter(lambda p: p.split('\t')[3] == 'test', pair_list))
        test_list = [i for i in test_list]
        # test_list = [i.strip() for i in open('../../input/MPV/clean_test.txt', 'r').readlines()]
        # train_list = train_list[:256]
        # test_list = train_list
        if istrain:
            self.mode = 'train'
            self.img_list = train_list
        else:
            self.mode = 'val'
            self.img_list = test_list
        self.train_mode = train_mode

        pose_list = [i.strip().split('.')[0] + '.txt' for i in open('demo/train.txt', 'r').readlines()]
        self.img_list = self.img_list * len(pose_list)
        for i in range(len(self.img_list)):
            self.img_list[i] = self.img_list[i] + '\t' + pose_list[i]
            
        # self.pose = open('../input/pose/' + self.mode + '/' + self.img_list).readline().split()
        # self.pose = self.pose2loc(self.pose)

    def __getitem__(self, index):
        try:
            img_source = self.img_list[index].split('\t')[0]
            img_target = self.img_list[index].split('\t')[1]
            cloth_img = self.img_list[index].split('\t')[2]
        except:
            img_source = self.img_list[index].split(' ')[0]
            img_target = self.img_list[index].split(' ')[1]
            cloth_img = self.img_list[index].split(' ')[2]

        source_splitext = os.path.splitext(img_source)[0]
        target_splitext = os.path.splitext(img_target)[0]
        cloth_splitext = os.path.splitext(cloth_img)[0]


                
        cloth_img_path = os.path.join('../../input/MPV' , self.mode, cloth_img)
        cloth = Image.open(cloth_img_path)
        cloth_image = self.transforms(cloth)


        if self.train_mode == 'parsing':

            cloth_img_path = os.path.join('../../input/MPV/parsing', self.mode, cloth_splitext + '.png')
            cloth = Image.open(cloth_img_path)
            cloth_array = np.array(cloth)
            cloth = (cloth_array > 0).astype(np.float32)
            cloth = torch.from_numpy(cloth)
            cloth = torch.unsqueeze(cloth, 0) # [1,256,192]

            source_img_path = os.path.join('../../input/MPV/parsing' , self.mode, source_splitext + '_vis.png')
            target_img_path = os.path.join('../../input/MPV/parsing' , self.mode, target_splitext + '_vis.png')
            source = np.array(Image.open(source_img_path))
            target = np.array(Image.open(target_img_path))

        elif self.train_mode == 'appearance' or self.train_mode == 'refine_cloth' or self.train_mode == 'refine_face':
            source_img_path = os.path.join('../../input/MPV', self.mode, source_splitext + '.jpg')
            target_img_path = os.path.join('../../input/MPV', self.mode, target_splitext + '.jpg')
            source = Image.open(source_img_path)
            target = Image.open(target_img_path)
            source = self.transforms(source)
            target = self.transforms(target) 



            cloth_img_path = os.path.join('../../input/MPV/parsing', self.mode, cloth_splitext + '.png')
            cloth = Image.open(cloth_img_path)
            cloth_array = np.array(cloth)
            cloth = (cloth_array > 0).astype(np.float32)
            cloth = torch.from_numpy(cloth)
            cloth_parse = torch.unsqueeze(cloth, 0) # [1,256,192]
        
        source_parse_path = os.path.join('../../input/MPV/parsing', self.mode, source_splitext + '.png')
        target_parse_path = os.path.join('demo/processed/787855/resize/parsing_png/resize/', self.img_list[index].split('\t')[4].split('.')[0] + '.png')
        source_parse = pose_utils.parsing_embedding(source_parse_path)
        target_parse = pose_utils.parsing_embedding(target_parse_path)

        self.size = source_parse.shape[1:3]

        target_pose = self.img_list[index].split('\t')[4]
        target_pose_path = os.path.join('demo/processed/787855/resize/pose/resize/', target_pose)
        
        target_pose = open(target_pose_path, 'r').readline().split()
        target_pose_loc = pose_utils.pose2loc(target_pose)
        scale = 1.2
        target_pose_loc = [[item[0] , int(item[1] * scale - 192 / 2 * (scale - 1))] for item in target_pose_loc]
        target_pose_embedding = pose_utils.heatmap_embedding(self.size, target_pose_loc)

        result = {
                'source_parse': source_parse,
                'target_parse': target_parse,
                'source_parse_vis': source_parse_vis,
                'target_parse_vis': target_parse_vis,
                'source_pose_embedding': source_pose_embedding,
                'target_pose_embedding': target_pose_embedding,
                'source_pose_map': source_pose_map.transpose(2,0,1),
                'target_pose_map': target_pose_map.transpose(2,0,1),
                'source_image': source_img, 
                'target_image': target_img,
                'cloth_image' : cloth_img,
                'cloth_parse' : cloth_parse,
                'interpol_pose_map': interpol_pose_map,
                'interpol_warps': interpol_warps,
                'interpol_masks': interpol_masks,
                'warps' : warps,
                'masks': masks,
                # downsample
                'downsample_source_img': downsample_source_img,
                'downsample_target_img': downsample_target_img,
                'downsample_source_pose_loc': downsample_source_pose_loc,
                'downsample_target_pose_loc': downsample_target_pose_loc,
                'downsample_source_pose_embedding': downsample_source_pose_embedding,
                'downsample_target_pose_embedding': downsample_target_pose_embedding,
                'downsample_source_pose_map': downsample_source_pose_map,
                'downsample_target_pose_map': downsample_target_pose_map,
                'downsample_source_parse': downsample_source_parse,
                'downsample_target_parse': downsample_target_parse,
                'downsample_warps': downsample_warps,
                'downsample_masks': downsample_masks,
                'source_parse_shape': source_parse_shape,
                'im_h': im_h, # source image head and hair
                'im_c': im_c # target_cloth_image
        }

        return result

    def __len__(self):
        return len(self.img_list)

if __name__ == '__main__':
    transforms = transforms.Compose([
                            transforms.ToTensor()
    ])
    train_mode = 'parsing'
    dataset = ClothDataset(True, transforms, train_mode)





import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os.path as osp
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision import utils
from utils import pose_utils
from PIL import ImageDraw
from .transforms import create_part

np.seterr(divide='ignore', invalid='ignore')

class ParseDataset(Dataset):
    def __init__(self, istrain, augment, train_mode):
        self.transforms = augment
        if istrain:
            self.mode = 'train'
            self.img_list = [i.strip() for i in open('../../input/train_pair.txt', 'r').readlines()]
        else:
            self.mode = 'val'
            self.img_list = [i.strip() for i in open('../../input/validation_pair.txt', 'r').readlines()]
        self.train_mode = train_mode

    def __getitem__(self, index):
        img_source = self.img_list[index].split('\t')[0]
        img_target = self.img_list[index].split('\t')[1]
        
        if self.train_mode == 'parsing':
            source_img_path = os.path.join('../../input/parsing' , self.mode, img_source.split('.')[0] + '_vis.png')
            target_img_path = os.path.join('../../input/parsing' , self.mode, img_target.split('.')[0] + '_vis.png')
            source = np.array(Image.open(source_img_path))
            target = np.array(Image.open(target_img_path))

        elif self.train_mode == 'appearance':
            source_img_path = os.path.join('../../input', self.mode, 'images', img_source.split('.')[0] + '.png')
            target_img_path = os.path.join('../../input', self.mode, 'images', img_target.split('.')[0] + '.png')
            source = Image.open(source_img_path)
            target = Image.open(target_img_path)
            source = self.transforms(source)
            target = self.transforms(target)

        source_parse_path = os.path.join('../../input/parsing' , self.mode, img_source.split('.')[0] + '.png')
        target_parse_path = os.path.join('../../input/parsing' , self.mode, img_target.split('.')[0] + '.png')
        source_parse = pose_utils.parsing_embedding(source_parse_path)
        target_parse = pose_utils.parsing_embedding(target_parse_path)

        self.size = source_parse.shape[1:3]
        
        target_pose_path = os.path.join('../../input/pose', self.mode, img_target.split('.')[0]+'.txt')
        target_pose = open(target_pose_path, 'r').readline().split()
        target_pose_loc = pose_utils.pose2loc(target_pose)
        target_pose_embedding = pose_utils.heatmap_embedding(self.size, target_pose_loc)
        # only for unified
        cloth = torch.Tensor([1,2])

        result = {
                'source_parse': source_parse,
                'target_pose_embedding': target_pose_embedding,
                'target_parse': target_parse,
                'source_image': source,
                'target_image': target,
                'cloth_image': cloth
        }
        
    def __len__(self):
        return len(self.img_list)


class ClothDataset(Dataset):
    def __init__(self, istrain, augment, train_mode):
        self.transforms = augment
        pair_list =[i.strip() for i in open('../../input/MPV/poseA_poseB_clothes.txt', 'r').readlines()] 
        # pair_list =[i.strip() for i in open('../../input/MPV/selected_pair.txt', 'r').readlines()] 
        train_list = list(filter(lambda p: p.split('\t')[3] == 'train', pair_list))
        train_list = [i for i in train_list]
        test_list = list(filter(lambda p: p.split('\t')[3] == 'test', pair_list))
        test_list = [i for i in test_list]
        # test_list = [i.strip() for i in open('../../input/MPV/clean_test.txt', 'r').readlines()]
        # train_list = train_list[:256]
        # test_list = train_list
        if istrain:
            self.mode = 'train'
            self.img_list = train_list
        else:
            self.mode = 'val'
            self.img_list = test_list
        
        self.train_mode = train_mode            
        # self.pose = open('../input/pose/' + self.mode + '/' + self.img_list).readline().split()
        # self.pose = self.pose2loc(self.pose)
        self.fine_width = 192
        self.fine_height = 256
        self.size = (256, 192)
    def __getitem__(self, index):

        try:
            img_source = self.img_list[index].split('\t')[0]
            img_target = self.img_list[index].split('\t')[1]
            cloth_img = self.img_list[index].split('\t')[2]
        except:
            img_source = self.img_list[index].split(' ')[0]
            img_target = self.img_list[index].split(' ')[1]
            cloth_img = self.img_list[index].split(' ')[2]

        source_splitext = os.path.splitext(img_source)[0]
        target_splitext = os.path.splitext(img_target)[0]
        cloth_splitext = os.path.splitext(cloth_img)[0]

        cloth_parse_path = os.path.join('../../input/MPV/parsing', self.mode, cloth_splitext + '.png')
        cloth_parse = Image.open(cloth_parse_path)
        cloth_parse_array = np.array(cloth_parse)
        cloth_parse = (cloth_parse_array > 0).astype(np.float32)
        cloth_parse = torch.from_numpy(cloth_parse)
        cloth_parse = torch.unsqueeze(cloth_parse, 0) # [1,256,192]

        cloth_img_path_ = os.path.join('../../input/MPV' , self.mode, cloth_img)
        cloth_img = Image.open(cloth_img_path_)
        cloth_img = self.transforms(cloth_img)

        source_img_path = os.path.join('../../input/MPV', self.mode, source_splitext + '.jpg')
        target_img_path = os.path.join('../../input/MPV', self.mode, target_splitext + '.jpg')
        source_img = Image.open(source_img_path)
        target_img = Image.open(target_img_path)

        downsample_source_img = source_img.resize((96,128),Image.BICUBIC)
        downsample_target_img = target_img.resize((96,128),Image.BICUBIC)

        downsample_source_img = self.transforms(downsample_source_img)
        downsample_target_img = self.transforms(downsample_target_img)

        source_img = self.transforms(source_img) # [-1, 1]
        target_img = self.transforms(target_img) # [-1, 1]

        # parsing

        source_parse_vis_path = os.path.join('../../input/MPV/parsing' , self.mode, source_splitext + '_vis.png')
        target_parse_vis_path = os.path.join('../../input/MPV/parsing' , self.mode, target_splitext + '_vis.png')
        source_parse_vis = self.transforms(Image.open(source_parse_vis_path)) # source_parse_vis
        target_parse_vis = self.transforms(Image.open(target_parse_vis_path))
        
        source_parse_path = os.path.join('../../input/MPV/parsing', self.mode, source_splitext + '.png')
        target_parse_path = os.path.join('../../input/MPV/parsing', self.mode, target_splitext + '.png')
        source_parse = pose_utils.parsing_embedding(source_parse_path)
        target_parse = pose_utils.parsing_embedding(target_parse_path)

        source_parse_shape = Image.fromarray((np.array(Image.open(source_parse_path))*255).astype(np.uint8))
        source_parse_shape = source_parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR) # downsample and then upsample
        source_parse_shape = source_parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        
        source_parse_shape = self.transforms(source_parse_shape) # [-1,1]

        ""
        source_parse_head = (np.array(Image.open(source_parse_path)) == 1).astype(np.float32) + \
                    (np.array(Image.open(source_parse_path)) == 2).astype(np.float32) + \
                    (np.array(Image.open(source_parse_path)) == 4).astype(np.float32) + \
                    (np.array(Image.open(source_parse_path)) == 13).astype(np.float32)


        target_parse_cloth = (np.array(Image.open(target_parse_path)) == 5).astype(np.float32) + \
                (np.array(Image.open(target_parse_path)) == 6).astype(np.float32) + \
                (np.array(Image.open(target_parse_path)) == 7).astype(np.float32)

        phead = torch.from_numpy(source_parse_head) # [0,1]

        # for target cloth_parse
        pcm = torch.from_numpy(target_parse_cloth) # [0,1]
        # upper cloth
        im = target_img # [-1,1]
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts --> white same as GT ...
        im_h = source_img * phead - (1 - phead) # [-1,1], fill -1 for other parts, thus become black visual

        ""
        # pose heatmap embedding
        source_pose_path = os.path.join('../../input/MPV/pose', self.mode, source_splitext +'.txt')
        source_pose = open(source_pose_path, 'r').readline().split()
        source_pose_loc = pose_utils.pose2loc(source_pose)
        source_pose_embedding = pose_utils.heatmap_embedding(self.size, source_pose_loc)
        


        target_pose_path = os.path.join('../../input/MPV/pose', self.mode, target_splitext +'.txt')
        target_pose = open(target_pose_path, 'r').readline().split()
        target_pose_loc = pose_utils.pose2loc(target_pose)
        target_pose_embedding = pose_utils.heatmap_embedding(self.size, target_pose_loc)
        
        source_pose_map = pose_utils.cords_to_map(np.array(source_pose_loc), self.size, 6)
        target_pose_map = pose_utils.cords_to_map(np.array(target_pose_loc), self.size, 6)

        interpol_pose_map = pose_utils.compute_interpol_map(source_pose_map, target_pose_map)
        interpol_warps, interpol_masks = pose_utils.compute_interpol_cord_warp(source_pose_map, interpol_pose_map)
        interpol_pose_map = torch.from_numpy(np.concatenate(interpol_pose_map, axis=0)).float()

        # directly use warps and masks
        warps, masks = pose_utils.compute_cord_warp(np.array(source_pose_loc), np.array(target_pose_loc))
        warps = warps.astype(np.float32)
        masks = masks.astype(np.float32)


        # downsample heatmap embedding and parsing
        downsample_source_pose_loc = pose_utils.downsample_pose_array(np.array(source_pose_loc)) 
        downsample_target_pose_loc = pose_utils.downsample_pose_array(np.array(target_pose_loc))
        downsample_source_pose_embedding = pose_utils.heatmap_embedding((128,96), downsample_source_pose_loc)
        downsample_target_pose_embedding = pose_utils.heatmap_embedding((128,96), downsample_target_pose_loc)

        downsample_source_pose_map = pose_utils.cords_to_map(np.array(downsample_source_pose_loc), (128,96), 6).transpose(2,0,1)
        downsample_target_pose_map = pose_utils.cords_to_map(np.array(downsample_target_pose_loc), (128,96), 6).transpose(2,0,1)
        
        [X, Y] = np.meshgrid(range(0,192,2),range(0,256,2))
        
        downsample_source_parse = source_parse[:,Y,X]
        downsample_target_parse = target_parse[:,Y,X]
        downsample_warps, downsample_masks = pose_utils.compute_cord_warp(downsample_source_pose_loc, downsample_target_pose_loc, _image_size=(128,96))
        downsample_warps = downsample_warps.astype(np.float32)
        downsample_masks = downsample_masks.astype(np.float32)

        result = {
                'source_parse': source_parse,
                'target_parse': target_parse,
                'source_parse_vis': source_parse_vis,
                'target_parse_vis': target_parse_vis,
                'source_pose_embedding': source_pose_embedding,
                'target_pose_embedding': target_pose_embedding,
                'source_pose_map': source_pose_map.transpose(2,0,1),
                'target_pose_map': target_pose_map.transpose(2,0,1),
                'source_image': source_img, 
                'target_image': target_img,
                'cloth_image' : cloth_img,
                'cloth_parse' : cloth_parse,
                'interpol_pose_map': interpol_pose_map,
                'interpol_warps': interpol_warps,
                'interpol_masks': interpol_masks,
                'warps' : warps,
                'masks': masks,
                # downsample
                'downsample_source_img': downsample_source_img,
                'downsample_target_img': downsample_target_img,
                'downsample_source_pose_loc': downsample_source_pose_loc,
                'downsample_target_pose_loc': downsample_target_pose_loc,
                'downsample_source_pose_embedding': downsample_source_pose_embedding,
                'downsample_target_pose_embedding': downsample_target_pose_embedding,
                'downsample_source_pose_map': downsample_source_pose_map,
                'downsample_target_pose_map': downsample_target_pose_map,
                'downsample_source_parse': downsample_source_parse,
                'downsample_target_parse': downsample_target_parse,
                'downsample_warps': downsample_warps,
                'downsample_masks': downsample_masks,
                'source_parse_shape': source_parse_shape,
                'im_h': im_h, # source image head and hair
                'im_c': im_c # target_cloth_image
        }

        # print('the warps result is', warps.shape, masks.shape, downsample_warps.shape, downsample_masks.shape)  # [10,8] [10,256,192]
        # print('the interpol_warps result is', interpol_warps.shape, interpol_masks.shape) # [6,10,8] [6,10,256,192]
        return result

    def __len__(self):
        return len(self.img_list)

if __name__ == '__main__':

    augment = transforms.Compose([
                                # transforms.Resize(cfg.size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]) # change to [C, H, W]
    train_dataset = ClothDataset(istrain=True, augment=augment, train_mode='appearance')
    train_dataloader = DataLoader(
                        train_dataset,
                        shuffle=True,
                        drop_last=False,
                        num_workers=8,
                        batch_size=32,
                        pin_memory=True)
    for i, result in enumerate(train_dataloader):
        source_pose_embedding = result['source_pose_embedding'].float().cuda()
        target_pose_embedding = result['target_pose_embedding'].float().cuda()
        source_pose_map = result['source_pose_map']
        target_pose_map = result['target_pose_map']

        source_image = result['source_image'].float().cuda()
        target_image = result['target_image'].float().cuda()

        source_parse = result['source_parse'].float().cuda()
        target_parse = result['target_parse'].float().cuda()

        cloth_image = result['cloth_image'].float().cuda()
        cloth_parse = result['cloth_parse'].float().cuda()

        interpol_pose_map = result['interpol_pose_map'].float().cuda()
        interpol_warps = result['interpol_warps'].float().cuda()
        interpol_masks = result['interpol_masks'].float().cuda()
        warps = result['warps'].float().cuda()
        masks = result['masks'].float().cuda()

        downsample_source_pose_loc = result['downsample_source_pose_loc']
        downsample_target_pose_loc = result['downsample_target_pose_loc']
        downsample_source_parse = result['downsample_source_parse']
        downsample_target_parse = result['downsample_target_parse']
        
        downsample_source_pose_embedding = result['downsample_source_pose_embedding']
        downsample_target_pose_embedding = result['downsample_source_pose_embedding']
        downsample_source_pose_map = result['downsample_source_pose_map']
        downsample_target_pose_map = result['downsample_target_pose_map']

        downsample_source_img = result['downsample_source_img'].float().cuda()
        downsample_target_img = result['downsample_target_img'].float().cuda()

        downsample_warps = result['downsample_warps'].float().cuda()
        downsample_masks = result['downsample_masks'].float().cuda()

        print(warps.shape)
        print(masks.shape)

    