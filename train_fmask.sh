#!/bin/bash
model=$1
channel=16
num_cells=5

save_dir=$2
export TORCH_HOME=/home/trofim/MF_NAS_KD2;
echo $model;

CUDA_VISIBLE_DEVICES=$4 OMP_NUM_THREADS=4 python ./exps/NAS-Bench-201/main.py \
	--mode specific-${model} --save_dir ${save_dir} --max_node 4 \
	--datasets cifar10 \
	--use_less 1 \
	--splits   1 \
	--xpaths $TORCH_HOME/cifar.python \
	--channel ${channel} --num_cells ${num_cells} \
	--workers 4 \
	--seeds 111 \
    --kd $3 \
    --fmask $5
