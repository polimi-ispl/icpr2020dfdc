#!/usr/bin/env bash
DEVICE=0

echo ""
echo "-------------------------------------------------"
echo "| Train Xception on FFc23                       |"
echo "-------------------------------------------------"
# put your FF++ source directory path for the extracted faces and Dataframe and uncomment the following line
# FFPP_FACES_DIR=/your/dfdc/faces/directory
# FFPP_FACES_DF=/your/dfdc/faces/dataframe/path
python train_binclass.py \
--net Xception \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
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
# put your DFDC source directory path for the extracted faces and Dataframe and uncomment the following line
# DFDC_FACES_DIR=/your/dfdc/faces/directory
# DFDC_FACES_DF=/your/dfdc/faces/dataframe/path
python train_binclass.py \
--net Xception \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
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
echo "| Train EfficientNetB4 on FFc23                  |"
echo "-------------------------------------------------"
# put your FF++ source directory path for the extracted faces and Dataframe and uncomment the following line
# FFPP_FACES_DIR=/your/dfdc/faces/directory
# FFPP_FACES_DF=/your/dfdc/faces/dataframe/path
python train_binclass.py \
--net EfficientNetB4 \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
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
# put your DFDC source directory path for the extracted faces and Dataframe and uncomment the following line
# DFDC_FACES_DIR=/your/dfdc/faces/directory
# DFDC_FACES_DF=/your/dfdc/faces/dataframe/path
python train_binclass.py \
--net EfficientNetB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
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
# put your FF++ source directory path for the extracted faces and Dataframe and uncomment the following line
# FFPP_FACES_DIR=/your/dfdc/faces/directory
# FFPP_FACES_DF=/your/dfdc/faces/dataframe/path
python train_triplet.py \
--net EfficientNetB4 \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
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
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
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
# put your DFDC source directory path for the extracted faces and Dataframe and uncomment the following line
# DFDC_FACES_DIR=/your/dfdc/faces/directory
# DFDC_FACES_DF=/your/dfdc/faces/dataframe/path
python train_triplet.py \
--net EfficientNetB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
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
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
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
# put your FF++ source directory path for the extracted faces and Dataframe and uncomment the following line
# FFPP_FACES_DIR=/your/dfdc/faces/directory
# FFPP_FACES_DF=/your/dfdc/faces/dataframe/path
python train_binclass.py \
--net EfficientNetAutoAttB4 \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
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
# put your DFDC source directory path for the extracted faces and Dataframe and uncomment the following line
# DFDC_FACES_DIR=/your/dfdc/faces/directory
# DFDC_FACES_DF=/your/dfdc/faces/dataframe/path
python train_binclass.py \
--net EfficientNetAutoAttB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
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
# put your FF++ source directory path for the extracted faces and Dataframe and uncomment the following line
# FFPP_FACES_DIR=/your/dfdc/faces/directory
# FFPP_FACES_DF=/your/dfdc/faces/dataframe/path
python train_binclass.py \
--net EfficientNetAutoAttB4 \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
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
# put your DFDC source directory path for the extracted faces and Dataframe and uncomment the following line
# DFDC_FACES_DIR=/your/dfdc/faces/directory
# DFDC_FACES_DF=/your/dfdc/faces/dataframe/path
python train_binclass.py \
--net EfficientNetAutoAttB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
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
# put your FF++ source directory path for the extracted faces and Dataframe and uncomment the following line
# FFPP_FACES_DIR=/your/dfdc/faces/directory
# FFPP_FACES_DF=/your/dfdc/faces/dataframe/path
python train_triplet.py \
--net EfficientNetAutoAttB4 \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
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
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
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
# put your DFDC source directory path for the extracted faces and Dataframe and uncomment the following line
# DFDC_FACES_DIR=/your/dfdc/faces/directory
# DFDC_FACES_DF=/your/dfdc/faces/dataframe/path
python train_triplet.py \
--net EfficientNetAutoAttB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
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
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
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


# With the following commands you can use only a subset of the 32 default frames per video. Just append `-Xfpv` to the `traindb` parameter, where X is the number of frames to use.

echo ""
echo "-------------------------------------------------"
echo "| Train Xception on FFc23  (variable fpv)        |"
echo "-------------------------------------------------"
# put your FF++ source directory path for the extracted faces and Dataframe and uncomment the following line
# FFPP_FACES_DIR=/your/dfdc/faces/directory
# FFPP_FACES_DF=/your/dfdc/faces/dataframe/path
python train_binclass.py \
--net Xception \
--traindb ff-c23-720-140-140-5fpv \
--valdb ff-c23-720-140-140 \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
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
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
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
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
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
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
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
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
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
