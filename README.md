# Detailed Vitural Try-on
Code for Detailed virtual try-on under arbitrary poses.

![Virtual try-on results](./demo/forward/0.jpg)

# Requirements
* Pytorch(0.4.1)
* Torchvision(0.2.1)
* pillow
* tqdm
* numpy
* json
* Tensorflow

# Getting Started 
## Installation
* Clone this repo
```
git clone https://github.com/AIprogrammer/Detailed-virtual-try-on.git. 
cd Detailed-virtual-try-on
```
* Download pre-trained models from [Google Drive](https://drive.google.com/open?id=1RpKC71VJub7hgWJ8dC5AWINW6nmZofm4). Put the checkpoints in "./checkpoints"

## Demo 
A demo model is given, and we provide some samples in "./dataset/images". The triplets including source image, target pose, target cloth is provided in the "./demo/demo.txt".

# Training

## Download the dataset
* Download the MPV dataset from [Image-based Multi-pose Virtual Try On](http://47.100.21.47:9999/overview.php) and put the dataset in "./dataset/images/".
* Select the postive perspective images and create dataset split file 'data_pair.txt', and then put the dataset in "./dataset/".

## Preprocess the dataset.
* Pose keypoints. Use the [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), and put the keypoints file in "./dataset/pose_coco".
* Semantic parsing. Use the [CIHP_PGN](https://github.com/Engineering-Course/CIHP_PGN), and put the parsing results in "./dataset/parse_cihp".
* Cloth mask. You can use the "GrabCut" method to get the cloth mask, and put the cloth mask in "./dataset/cloth_mask".