#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
data=cifar10
arch='ConvBig'
num=100
pruning_rate=70
tv=0.0004
log=out_70_new_attack.txt


for id in $(seq 0 $num);
do
{
    python3 -u reconstruct_image.py --target_id=$id --model=$arch --defense=ours --pruning_rate=$pruning_rate --tv=$tv --save_imag >> $log
}
done
