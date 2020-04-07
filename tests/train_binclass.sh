#!/usr/bin/env bash
python train_binclass.py \
--net Xception \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 2 \
--patience 10 \
--maxiter 4 \
--logint 1 \
--seed 41 \
--attention \
--log_dir runs/binclass \
--models_dir weights/binclass \
--debug