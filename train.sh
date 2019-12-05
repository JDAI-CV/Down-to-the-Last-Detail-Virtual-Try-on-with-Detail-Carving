### Step 1: Train Clothing Spatial Alignment
python train.py --train_mode gmm
###To save the training time and memory usage, we warped the in-shop cloth in an offline manner using well-trained Clothing Spatial Alignment module. 
CUDA_VISIBLE_DEVICES=0 python demo.py --batch_size_v 1 --num_workers 4 --warp_cloth
### Step 2: Train Parsing Transformation
python train.py --train_mode parsing
### Step 3: Train Detailed Appearance Generation
python train.py --train_mode appearance --batch_size_t 16 --save_time --suffix refactor --val_freq 20 --save_epoch_freq 1 --joint_all --joint_parse_loss
### Step 4: Train Face Refinement
python train.py --train_mode face
### Step 5: Train jointly
python train.py --train_mode appearance --joint_all --joint_parse_loss