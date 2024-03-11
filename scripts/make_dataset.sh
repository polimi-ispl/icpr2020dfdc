#!/usr/bin/env bash

echo ""
echo "-------------------------------------------------"
echo "| Index DFDC dataset                            |"
echo "-------------------------------------------------"
# put your dfdc source directory path and uncomment the following line
# DFDC_SRC=/your/dfdc/train/split/source/directory
python index_dfdc.py --source $DFDC_SRC

echo ""
echo "-------------------------------------------------"
echo "| Index FF dataset                              |"
echo "-------------------------------------------------"
# put your ffpp source directory path and uncomment the following line
# FFPP_SRC=/your/ffpp/source/directory
python index_ffpp.py --source $FFPP_SRC


echo ""
echo "-------------------------------------------------"
echo "| Extract faces from DFDC                        |"
echo "-------------------------------------------------"
# put your source and destination directories and uncomment the following lines
# DFDC_SRC=/your/dfdc/source/folder
# VIDEODF_SRC=/previously/computed/index/path
# FACES_DST=/faces/output/directory
# FACESDF_DST=/faces/df/output/directory
# CHECKPOINT_DST=/tmp/per/video/outputs
python extract_faces.py \
--source $DFDC_SRC \
--videodf $VIDEODF_SRC \
--facesfolder $FACES_DST \
--facesdf $FACESDF_DST \
--checkpoint $CHECKPOINT_DST

echo ""
echo "-------------------------------------------------"
echo "| Extract faces from FF                         |"
echo "-------------------------------------------------"
# put your source and destination directories and uncomment the following lines
# FFPP_SRC=/your/dfdc/source/folder
# VIDEODF_SRC=/previously/computed/index/path
# FACES_DST=/faces/output/directory
# FACESDF_DST=/faces/df/output/directory
# CHECKPOINT_DST=/tmp/per/video/outputs
python extract_faces.py \
--source $FFPP_SRC \
--videodf $VIDEODF_SRC \
--facesfolder $FACES_DST \
--facesdf $FACESDF_DST \
--checkpoint $CHECKPOINT_DST
