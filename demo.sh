#! /bin/bash
echo "Running forward"
set -ex
CUDA_VISIBLE_DEVICES=0 python demo.py --batch_size_v 80 --num_workers 4 --forward_save_path 'demo/forward' --isdemo --save_time
