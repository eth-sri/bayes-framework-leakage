#!/bin/bash
python3 tune_attacks_dp.py \
	   -pa $1 \
	   --attack --dataset MNIST --network cnn --delta 20.0 --n_clients 1 \
	   --batch_size 1 -lr 0.001 --n_steps 100000 \
	   --defense_lr 0.01 --defense $2 --k_batches 10 --step_def_epochs 5 \
	   --attack_total_variation 1.0 --att_every 50 --n_attack 10 --att_init random --att_lr 0.5 --exp_decay_factor 0.94 --att_epochs 100 \
	   --reg_tv 0.25 --reg_clip 0.75 \
	   --att_metric l2 \
	   > $3_$2_l2_prior.txt 2>&1 


