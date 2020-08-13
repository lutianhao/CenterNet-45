#!/usr/bin/env bash
python train.py --log_name pascal-resdcn18_384_dp --dataset pascal --arch resdcn_18 --img_size 384 --lr 1.25e-4 --lr_step 45,60 --batch_size 32 --num_epochs 70 --num_workers 10
