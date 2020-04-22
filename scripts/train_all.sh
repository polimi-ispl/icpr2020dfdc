#!/usr/bin/env bash
DEVICE=0

echo ""
echo "-------------------------------------------------"
echo "| Train Xception on FFc23  (variable fpv)        |"
echo "-------------------------------------------------"
python train_binclass.py \
--net Xception \
--traindb ff-c23-720-140-140-5fpv \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE

python train_binclass.py \
--net Xception \
--traindb ff-c23-720-140-140-10fpv \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE

python train_binclass.py \
--net Xception \
--traindb ff-c23-720-140-140-15fpv \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE

python train_binclass.py \
--net Xception \
--traindb ff-c23-720-140-140-20fpv \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE

python train_binclass.py \
--net Xception \
--traindb ff-c23-720-140-140-25fpv \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE

echo ""
echo "-------------------------------------------------"
echo "| Train Xception on FFc23                       |"
echo "-------------------------------------------------"
python train_binclass.py \
--net Xception \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Train Xception on DFDC                         |"
echo "-------------------------------------------------"
python train_binclass.py \
--net Xception \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Train Xception on FFc23 (triplet)             |"
echo "-------------------------------------------------"
python train_triplet.py \
--net Xception \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 12 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 60000 \
--seed 41 \
--attention \
--embedding \
--device $DEVICE

python train_binclass.py \
--net XceptionST \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--device $DEVICE \
--init weights/triplet/net-Xception_traindb-ff-c23-720-140-140_face-scale_size-224_seed-41/bestval.pth


echo ""
echo "-------------------------------------------------"
echo "| Train Xception on DFDC (triplet)              |"
echo "-------------------------------------------------"
python train_triplet.py \
--net Xception \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 12 \
--lr 1e-6 \
--valint 500 \
--patience 10 \
--maxiter 60000 \
--seed 41 \
--attention \
--embedding \
--device $DEVICE

python train_binclass.py \
--net XceptionST \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--device $DEVICE \
--init weights/triplet/net-Xception_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth


echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetB4 on FFc23                  |"
echo "-------------------------------------------------"
python train_binclass.py \
--net EfficientNetB4 \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetB4 on DFDC                   |"
echo "-------------------------------------------------"
python train_binclass.py \
--net EfficientNetB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE

echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetB4 on FFc23 (triplet)        |"
echo "-------------------------------------------------"
python train_triplet.py \
--net EfficientNetB4 \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 12 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 60000 \
--seed 41 \
--attention \
--embedding \
--device $DEVICE

python train_binclass.py \
--net EfficientNetB4ST \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--device $DEVICE \
--init weights/triplet/net-EfficientNetB4_traindb-ff-c23-720-140-140_face-scale_size-224_seed-41/bestval.pth


echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetB4 on DFDC (triplet)        |"
echo "-------------------------------------------------"
python train_triplet.py \
--net EfficientNetB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 12 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 60000 \
--seed 41 \
--attention \
--embedding \
--device $DEVICE

python train_binclass.py \
--net EfficientNetB4ST \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--device $DEVICE \
--init weights/triplet/net-EfficientNetB4_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth


echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetAutoAttB4 on FFc23           |"
echo "-------------------------------------------------"
python train_binclass.py \
--net EfficientNetAutoAttB4 \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetAutoAttB4 on DFDC           |"
echo "-------------------------------------------------"
python train_binclass.py \
--net EfficientNetAutoAttB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE

echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetAutoAttB4 on FFc23 (tuning) |"
echo "-------------------------------------------------"
python train_binclass.py \
--net EfficientNetAutoAttB4 \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--init weights/binclass/net-EfficientNetB4_traindb-ff-c23-720-140-140_face-scale_size-224_seed-41/bestval.pth \
--suffix finetuning \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetAutoAttB4 on DFDC  (tuning) |"
echo "-------------------------------------------------"
python train_binclass.py \
--net EfficientNetAutoAttB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--init weights/binclass/net-EfficientNetB4_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth \
--suffix finetuning \
--device $DEVICE

echo ""
echo "---------------------------------------------------"
echo "| Train EfficientNetAutoAttB4AT on FFc23 (tuning) |"
echo "---------------------------------------------------"
python train_binclass.py \
--net EfficientNetAutoAttB4AT \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--init weights/binclass/net-EfficientNetB4_traindb-ff-c23-720-140-140_face-scale_size-224_seed-41/bestval.pth \
--suffix finetuning \
--device $DEVICE


echo ""
echo "---------------------------------------------------"
echo "| Train EfficientNetAutoAttB4AT on DFDC  (tuning) |"
echo "---------------------------------------------------"
python train_binclass.py \
--net EfficientNetAutoAttB4AT \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--init weights/binclass/net-EfficientNetB4_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth \
--suffix finetuning \
--device $DEVICE

echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetAutoAttB4a on FFc23         |"
echo "-------------------------------------------------"
python train_binclass.py \
--net EfficientNetAutoAttB4a \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetAutoAttB4a on DFDC          |"
echo "-------------------------------------------------"
python train_binclass.py \
--net EfficientNetAutoAttB4a \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE

echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetAutoAttB4a on FFc23 (tuning)|"
echo "-------------------------------------------------"
python train_binclass.py \
--net EfficientNetAutoAttB4a \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--init weights/binclass/net-EfficientNetB4_traindb-ff-c23-720-140-140_face-scale_size-224_seed-41/bestval.pth \
--suffix finetuning \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetAutoAttB4a on DFDC  (tuning)|"
echo "-------------------------------------------------"
python train_binclass.py \
--net EfficientNetAutoAttB4a \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--init weights/binclass/net-EfficientNetB4_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth \
--suffix finetuning \
--device $DEVICE

echo ""
echo "---------------------------------------------------"
echo "| Train EfficientNetAutoAttB4aAT on FFc23 (tuning)|"
echo "---------------------------------------------------"
python train_binclass.py \
--net EfficientNetAutoAttB4aAT \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--init weights/binclass/net-EfficientNetB4_traindb-ff-c23-720-140-140_face-scale_size-224_seed-41/bestval.pth \
--suffix finetuning \
--device $DEVICE


echo ""
echo "---------------------------------------------------"
echo "| Train EfficientNetAutoAttB4aAT on DFDC  (tuning)|"
echo "---------------------------------------------------"
python train_binclass.py \
--net EfficientNetAutoAttB4aAT \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--init weights/binclass/net-EfficientNetB4_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth \
--suffix finetuning \
--device $DEVICE

echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetAutoAttB4 on FFc23 (triplet)|"
echo "-------------------------------------------------"
python train_triplet.py \
--net EfficientNetAutoAttB4 \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 12 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 60000 \
--seed 41 \
--attention \
--embedding \
--device $DEVICE

python train_binclass.py \
--net EfficientNetAutoAttB4ST \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--device $DEVICE \
--init weights/triplet/net-EfficientNetAutoAttB4_traindb-ff-c23-720-140-140_face-scale_size-224_seed-41/bestval.pth


echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetAutoAttB4 on DFDC (triplet) |"
echo "-------------------------------------------------"
python train_triplet.py \
--net EfficientNetAutoAttB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 12 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 60000 \
--seed 41 \
--attention \
--embedding \
--device $DEVICE

python train_binclass.py \
--net EfficientNetAutoAttB4ST \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--device $DEVICE \
--init weights/triplet/net-EfficientNetAutoAttB4_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth
