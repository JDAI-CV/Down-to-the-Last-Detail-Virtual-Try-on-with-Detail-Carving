import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from skimage.morphology import square, dilation, erosion
from utils import pose_transform
# import pose_transform
import json
from PIL import ImageDraw
from torchvision import utils, transforms
from skimage.draw import circle, line_aa, polygon

np.seterr(divide='ignore', invalid='ignore')


LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [2,16], [5,17]]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
               'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

MISSING_VALUE = 0


LIMB_SEQ_PAF = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [2,16], [5,17]]


LABELS_PAF = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
               'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

label_colours = [(0,0,0)
                # 0=Background
                ,(128,0,0),(255,0,0),(0,85,0),(170,0,51),(255,85,0)
                # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
                ,(0,0,85),(0,119,221),(85,85,0),(0,85,85),(85,51,0)
                # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
                ,(52,86,128),(0,128,0),(0,0,255),(51,170,221),(0,255,255)
                # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
                ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
                # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe

# circle embedding
def heatmap_embedding(size, pose, radius=6):
    h, w = size # 128,96
    emb_all = np.empty((1, h, w)).astype(np.uint8)
    for item in pose:
        emb = np.zeros((1, h, w))
        x, y = item
        if x > 1 and y > 1:
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    distance = np.sqrt(float(i**2 + j**2))
                    if x+i >= 0 and x+i < h and y+j >= 0 and y+j < w:
                        if distance <= radius:
                            emb[:, x+i, y+j] = 1
        emb_all = np.concatenate((emb_all, emb), axis=0)
    return emb_all[1:,:,:]

def parsing_embedding(parse_path):
    parse = Image.open(parse_path) # 256,192
    parse = np.array(parse)
    parse_emb = []
    for i in range(20):
        parse_emb.append((parse == i).astype(np.float32).tolist())
    parse = np.array(parse_emb).astype(np.float32)
    return parse

def pose2loc(pose):
    x = []
    y = []
    loc = []

    for index, item in enumerate(pose):
        if index % 3 == 0:
            x.append(int(item))
        elif (index + 2) % 3 == 0:
            y.append(int(item))
    for i,j in zip(x, y):
        if i > 191:
            i = 191
        if j > 255:
            j = 255
        loc.append([j, i])# attention the indices 
    return loc

def draw_pose_from_cords(pose_joints, img_size, radius=6, draw_joints=True):

    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    mask = np.zeros(shape=img_size, dtype=bool)   
    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            yy, xx = np.minimum(yy, 255), np.minimum(xx, 191)
            colors[yy, xx] = np.expand_dims(val, 1) * 255
            mask[yy, xx] = True

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]
        mask[yy, xx] = True

    return colors, mask

def decode_labels(mask, num_images=1, num_classes=20):
    n, h, w, c = mask.shape  # batch_size, 256, 256, 1
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_,j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

def array2img(img_array):
    imgs = (img_array.permute(0,2,3,1).contiguous().cpu().numpy() * 0.5 + 0.5) * 255
    imgs = imgs.astype(np.uint8)
    return imgs

# all parsing_vis and mask can not use tensor type
def save_img(images, path):
    img = []
    assert len(images) > 0

    for i in range(len(images)):
        if isinstance(images[i], np.ndarray):
            if images[i].shape[3] == 1:
                images[i] = images[i].repeat(3, axis=3)
            elif images[i].shape[3] != 3:
                images[i] = images[i].transpose((0,2,3,1))
        else:
            if images[i].shape[1] == 1:
                images[i] = images[i].repeat(1,3,1,1).cpu().numpy()
            elif images[i].shape[1] == 3:
                images[i] = ((images[i].permute(0,2,3,1).contiguous().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
            else:
                images[i] = images[i].cpu().numpy()

    for i in range(len(images[0])):
        img.append(np.concatenate([image[i] for image in images], axis=1))
    
    img = np.concatenate(img, axis=0)

    image = Image.fromarray(img.astype(np.uint8))
    image.save(path)

def save_img_grid(images, path):
    assert len(images) > 0
    for i in range(len(images)):
        if isinstance(images[i], np.ndarray):
            images[i] = torch.from_numpy(images[i].astype(np.float32)).permute(0,3,1,2).contiguous()
            images[i] = ((images[i] / 255) - 0.5) / 0.5
                    
        else:
            images[i] = images[i].cpu()
    image_vis = []

    for i in range(len(images[0])):
        img = torch.cat([image[i] for image in images], dim=2)

        image_vis.append(img.unsqueeze(dim=0))
    
    image_vis = [img.cpu() for img in image_vis]
    image_vis = torch.cat(image_vis, dim=0) * 0.5 + 0.5

    utils.save_image(image_vis, path, 4, padding=5)


# based on keypoint --> radius heatmap of single keypoint
def _getSparseKeypoint(r, c, k, height, width, radius=4, var=4, mode='Solid'):
    # r, c is the x/y coordination of keypoint, k is the num of keypoints
    r = int(r)
    c = int(c)
    k = int(k)
    indices = []
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            distance = np.sqrt(float(i**2 + j**2))
            if r+i >= 0 and r+i < height and c+j >= 0 and c+j < width:
                if 'Solid' == mode and distance <= radius:
                    indices.append([r+i, c+j, k])

    return indices


# peaks all keypoints
def _getSparsePose(peaks, height, width, channel, radius=4, var=4, mode='Solid'):
    indices = []
    values = []
    for k in range(len(peaks)):
        p = peaks[k]
        if 0!=len(p):
            r = p[0][1]
            c = p[0][0]
            ind = _getSparseKeypoint(r, c, k, height, width, radius, var, mode)
            indices.extend(ind)
    # indices [num_keypoint, height, width]
    shape = [height, width, channel]
    return indices, shape


# 1 else zero
def _sparse2dense(indices, shape):
    dense = np.zeros(shape)
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        k = indices[i][2]
        dense[r,c,k] = 1
    return dense


# base on the heatmap do erosion to get the mask
def _getPoseMask(peaks, height, width, radius=4, var=4, mode='Solid'):
    limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
                         [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
                         [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]]
    indices = []
    for limb in limbSeq:
        p0 = peaks[limb[0] -1]
        p1 = peaks[limb[1] -1]
        if 0!=len(p0) and 0!=len(p1):
            r0 = p0[1]
            c0 = p0[0]
            r1 = p1[1]
            c1 = p1[0]
            ind  = _getSparseKeypoint(r0, c0, 0, height, width, radius, var, mode)
            indices.extend(ind)
            ind = _getSparseKeypoint(r1, c1, 0, height, width, radius, var, mode)
            indices.extend(ind)
            
            distance = np.sqrt((r0-r1)**2 + (c0-c1)**2)
            sampleN = int(distance/radius)
            if sampleN > 1:
                for i in range(1,sampleN):
                    r = r0 + (r1-r0)*i/sampleN
                    c = c0 + (c1-c0)*i/sampleN
                    ind = _getSparseKeypoint(r, c, 0, height, width, radius, var, mode)
                    indices.extend(ind)
                    
    shape = [height, width, 1]
    
    dense = np.squeeze(_sparse2dense(indices, shape))
    dense = dilation(dense, square(5))
    dense = erosion(dense, square(5))
    return dense


def cords_to_map(cords, img_size, sigma=100):
    MISSING_VALUE = 0
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))   # 192, 256
        result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
    return result


def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)


# gaussian map to cordinate
def map_to_cord(pose_map, pose_dim, threshold=0.1):
    MISSING_VALUE = 0
    all_peaks = [[] for i in range(pose_dim)]
    pose_map = pose_map[..., :pose_dim]
    
    y, x, z = np.where(np.logical_and(pose_map == pose_map.max(axis = (0, 1)),
                                     pose_map > threshold))
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(pose_dim):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(MISSING_VALUE)
            y_values.append(MISSING_VALUE)

    return np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)


def compute_pose_map(self, pair, direction):
    assert direction in ['to', 'from']
    
    pose_map = np.empty(list(self._image_size) + [self.pose_dim])
    row = self._annotations_file.loc[pair[direction]]
    # file_name = self._tmp_pose + pair[direction] + '.npy'
    # if os.path.exists(file_name):
    #     pose = np.load(file_name)
    # else:

    # pose cordinate
    kp_array = load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])


    pose_map = cords_to_map(kp_array, self._image_size)
    # np.save(file_name, pose)
    # channels must be first
    return np.transpose(pose_map, [2,0,1])



def compute_interpol_map(inp_map, tg_map):
    
    # map to cord expects channels in last dim (old keras code)
    num_stacks = 5
    pose_dim = 18
    _image_size = (256,192)
    # source_cord = pose_utils.map_to_cord(source_map, 16, 0.1)
    inp_pos = map_to_cord(inp_map, pose_dim)
    tg_pos = map_to_cord(tg_map, pose_dim)
    pose_maps = []
    # compute interpol poses equal to num_stacks, with final pose being equal to the target psoe
    for i in range(1,num_stacks+1):
        interpol_pose = compute_interpol_pose(inp_pos,tg_pos,i, num_stacks, pose_dim)
        # print(interpol_pose)
        interpol_pose_map = cords_to_map(interpol_pose, _image_size)
        pose_maps.append(np.transpose(interpol_pose_map, [2,0,1]))
    return pose_maps


def compute_interpol_cord_warp(inp_map, interpol_pose):
    interpol_pose = [inp_map] + interpol_pose
    num_interpol = len(interpol_pose)
    interpol_warps, interpol_masks = [], []
    _warp_skip = 'mask'
    pose_dim = 18
    _image_size = (256,192)
    # possibly causing nan error
    kp_array1 = map_to_cord(inp_map, pose_dim)
    
    for pose in interpol_pose:
        if _warp_skip == 'full':
            warp = [np.empty([1, 8]), 1]
        else:
            warp = [np.empty([10, 8]),
                    np.empty([10] + list(_image_size))]
        kp_array2 = map_to_cord(pose, pose_dim)

        if _warp_skip == 'mask':
            warp[0] = pose_transform.affine_transforms(kp_array1, kp_array2, pose_dim)
            warp[1] = pose_transform.pose_masks(kp_array2, _image_size, pose_dim)
        else:
            warp[0] = pose_transform.estimate_uniform_transform(kp_array1, kp_array2, pose_dim)
        interpol_warps.append(warp[0])
        interpol_masks.append(warp[1])

        kp_array1 = kp_array2
        
    return np.array(interpol_warps), np.array(interpol_masks)


def compute_interpol_pose(inp_pos,tg_pos,index,num_stacks,pose_dim):
    assert index <= num_stacks
    if(pose_dim==16):
        interpol_pose = inp_pos + (tg_pos-inp_pos)*index/num_stacks
    # bad logic to circumvent missing annot . . synthesize and vanish missing annot after half sequence is completed
    elif(pose_dim==18):                                                                                                                   
        interpol_pose = np.zeros([pose_dim,2], dtype='float32')
        for i in range(pose_dim):
            # inp pose has missing annot and tg pose has it
            if ((inp_pos[i,0] == MISSING_VALUE or inp_pos[i,1]== MISSING_VALUE) and
                    (tg_pos[i,0] != MISSING_VALUE and tg_pos[i,1]!= MISSING_VALUE)):
                if(index<=num_stacks//2):
                    interpol_pose[i] = MISSING_VALUE
                else:
                    interpol_pose[i] = tg_pos[i]
            # tg pose has missing annot and inp pose has it
            elif ((tg_pos[i, 0] == MISSING_VALUE or tg_pos[i, 1] == MISSING_VALUE) and (
                    inp_pos[i, 0] != MISSING_VALUE and inp_pos[i, 1] != MISSING_VALUE)):
                if (index <= num_stacks // 2):
                    interpol_pose[i] = inp_pos[i]
                else:
                    interpol_pose[i] = MISSING_VALUE
            # annot missing in both poses
            elif ((tg_pos[i, 0] == MISSING_VALUE or tg_pos[i, 1] == MISSING_VALUE) and (
                    inp_pos[i, 0] == MISSING_VALUE or inp_pos[i, 1] == MISSING_VALUE)):
                interpol_pose[i] = MISSING_VALUE
            # normal interpol when annot are present in both cases
            else:
                interpol_pose[i] = inp_pos[i] + (tg_pos[i]-inp_pos[i])*index/num_stacks
    return interpol_pose


def compute_cord_warp(source_pose_loc, target_pose_loc, _warp_skip = 'mask', _image_size=(256,192), pose_dim=16):
    
    _image_size = (256,192)
    if _warp_skip == 'full':
        # warp = [np.empty([1, pose_dim-10]), 1]
        warp = [np.empty([1,8]), 1]
    else:
        # warp = [np.empty([10, pose_dim-10]),
        #         np.empty([10] + list(_image_size))]
        warp = [np.empty([10, 8]),
            np.empty([10] + list(_image_size))]

    kp_array1 = source_pose_loc
    kp_array2 = target_pose_loc

    if _warp_skip == 'mask':
        warp[0] = pose_transform.affine_transforms(kp_array1, kp_array2, pose_dim)
        warp[1] = pose_transform.pose_masks(kp_array2, _image_size, pose_dim)
    else:
        warp[0] = pose_transform.estimate_uniform_transform(kp_array1, kp_array2, pose_dim)
    return warp

def downsample_pose_array(input_array):
    kp_array = input_array.copy()
    for idx in range(18):
        if kp_array[idx,0] != -1 and kp_array[idx,1] != -1:
            kp_array[idx,0] = int(kp_array[idx,0] * 0.5)
            kp_array[idx,1] = int(kp_array[idx,1] * 0.5)

    return kp_array

if __name__ == '__main__':
    pass
    








    

