#!/usr/bin/env bash
DEVICE=0

echo ""
echo "-------------------------------------------------"
echo "| Test Xception on FFc23                        |"
echo "-------------------------------------------------"
# put your DFDC source directory path for the extracted faces and Dataframe and uncomment the following lines
# DFDC_FACES_DIR=/your/dfdc/faces/directory
# DFDC_FACES_DF=/your/dfdc/faces/dataframe/path
# put your FF++ source directory path for the extracted faces and Dataframe and uncomment the following lines
# FFPP_FACES_DIR=/your/dfdc/faces/directory
# FFPP_FACES_DF=/your/dfdc/faces/dataframe/path
python test_model.py \
--model_path weights/binclass/net-Xception_traindb-ff-c23-720-140-140_face-scale_size-224_seed-41/bestval.pth \
--testsets ff-c23-720-140-140 dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Test Xception on DFDC                         |"
echo "-------------------------------------------------"
python test_model.py \
--model_path weights/binclass/net-Xception_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth \
--testsets ff-c23-720-140-140 dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Test EfficientNetB4 on FFc23                  |"
echo "-------------------------------------------------"
python test_model.py \
--model_path weights/binclass/net-EfficientNetB4_traindb-ff-c23-720-140-140_face-scale_size-224_seed-41/bestval.pth \
--testsets ff-c23-720-140-140 dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Test EfficientNetB4 on DFDC                   |"
echo "-------------------------------------------------"
python test_model.py \
--model_path weights/binclass/net-EfficientNetB4_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth \
--testsets ff-c23-720-140-140 dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Test EfficientNetB4 on FFc23 (triplet)        |"
echo "-------------------------------------------------"
python test_model.py \
--model_path weights/binclass/net-EfficientNetB4ST_traindb-ff-c23-720-140-140_face-scale_size-224_seed-41/bestval.pth \
--testsets ff-c23-720-140-140 dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Test EfficientNetB4 on DFDC (triplet)         |"
echo "-------------------------------------------------"
python test_model.py \
--model_path weights/binclass/net-EfficientNetB4ST_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth \
--testsets ff-c23-720-140-140 dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Test EfficientNetAutoAttB4 on FFc23           |"
echo "-------------------------------------------------"
python test_model.py \
--model_path weights/binclass/net-EfficientNetAutoAttB4_traindb-ff-c23-720-140-140_face-scale_size-224_seed-41/bestval.pth \
--testsets ff-c23-720-140-140 dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Test EfficientNetAutoAttB4 on DFDC            |"
echo "-------------------------------------------------"
python test_model.py \
--model_path weights/binclass/net-EfficientNetAutoAttB4_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth \
--testsets ff-c23-720-140-140 dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Test EfficientNetAutoAttB4 on FFc23 (triplet) |"
echo "-------------------------------------------------"
python test_model.py \
--model_path weights/binclass/net-EfficientNetAutoAttB4ST_traindb-ff-c23-720-140-140_face-scale_size-224_seed-41/bestval.pth \
--testsets ff-c23-720-140-140 dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Test EfficientNetAutoAttB4 on DFDC (triplet)  |"
echo "-------------------------------------------------"
python test_model.py \
--model_path weights/binclass/net-EfficientNetAutoAttB4ST_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth \
--testsets ff-c23-720-140-140 dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
--device $DEVICE