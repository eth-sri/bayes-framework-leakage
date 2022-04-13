#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
data=cifar10
arch='ConvBig'
num=100
pruning_rate=80
tv=0.0004
log=log.txt
# Normally used steps  1 2 5 10 20 50 100 200 500 1000 2000 5000 10000
# NOTE: Don't start too many parallel experiments at once - this script makes them hard to kill!!!

for ctr in 10 20 50;
do
{
    python3 -u reconstruct_image_set.py --model=$arch --defense=ours --pruning_rate=$pruning_rate --tv=$tv --save_imag --steps=$ctr --use_steps &
}
done