#!/usr/bin/env bash
python train_triplet.py \
--net EfficientNetAutoAttB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 12 \
--lr 1e-5 \
--valint 2 \
--patience 10 \
--maxiter 4 \
--logint 1 \
--seed 41 \
--attention \
--embedding \
--embeddingint 2 \
--log_dir runs/triplet \
--models_dir weights/triplet \
--debug