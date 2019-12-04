#Detailed Vitural Try-on
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
git clone https://github.com/AIprogrammer/Detailed-virtual-try-on.git
cd Detailed-virtual-try-on

* Download pre-trained models from Google Drive(https://drive.google.com/open?id=1RpKC71VJub7hgWJ8dC5AWINW6nmZofm4). Put the checkpoints in "./checkpoints"

## demo 
A demo model is given, and we provide some samples in "./dataset/images". The triplets including source image, target pose, target cloth is provided in the "./demo/demo.txt".


# Data Preparation
We have provide the dataset split file in ./dataset/data_pair.txt

## Download the dataset
* Download the MPV dataset from Image-based Multi-pose Virtual Try On(http://47.100.21.47:9999/overview.php). You need to put the dataset in ./dataset/images/.
## Getting the pose keypoints files, semantic parsing, cloth mask
* Pose keypoints. Use the Openpose(https://github.com/CMU-Perceptual-Computing-Lab/openpose), and put the keypoints file in "./dataset/pose_coco".
* Semantic parsing. Use the CIHP_PGN(https://github.com/Engineering-Course/CIHP_PGN), and put the parsing results in "./dataset/parse_cihp".
* Cloth mask. You can use the "GrabCut" method to process the in-shop cloth to get the cloth mask, and put the cloth mask in "./dataset/cloth_mask".




