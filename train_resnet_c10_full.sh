#!/bin/bash
model='|nor_conv_3x3~0|+|none~0|nor_conv_3x3~1|+|skip_connect~0|none~1|skip_connect~2|'
channel=16
num_cells=5

save_dir=./output/NAS-BENCH-201-4/
export TORCH_HOME=/home/trofim/MF_NAS_KD2;

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 python ./exps/NAS-Bench-201/main.py \
	--mode specific-${model} --save_dir ${save_dir} --max_node 4 \
	--datasets cifar10 \
	--use_less 0 \
	--splits   1 \
    --xpaths $TORCH_HOME/cifar.python \
	--channel ${channel} --num_cells ${num_cells} \
	--workers 4 \
	--seeds 777

