#!/usr/bin/env bash
DEVICE=0

echo ""
echo "-------------------------------------------------"
echo "| Index DFDC dataset                            |"
echo "-------------------------------------------------"
python index_dfdc.py

echo ""
echo "-------------------------------------------------"
echo "| Index FF dataset                              |"
echo "-------------------------------------------------"
python index_dfdc.py


echo ""
echo "-------------------------------------------------"
echo "| Extract faces from DFDC                        |"
echo "-------------------------------------------------"
python extrac_faces.py \
--source dataset/dfdc_train_all/ \
--facesfolder data/facecache/dfdc_train_all/ \
--videodf data/dfdc_videos.pkl \
--facesdf data/dfdc_faces.pkl \
--checkpoint tmp/dfdc_prep/

echo ""
echo "-------------------------------------------------"
echo "| Extract faces from DFDC                        |"
echo "-------------------------------------------------"
python extract_faces.py \
--source dataset/dfdc_train_all/ \
--facesfolder data/facecache/dfdc_train_all/ \
--videodf data/dfdc_videos.pkl \
--facesdf data/dfdc_faces.pkl \
--checkpoint tmp/dfdc_prep/

echo ""
echo "-------------------------------------------------"
echo "| Extract faces from FF                         |"
echo "-------------------------------------------------"
python extract_faces.py \
--source dataset/ffpp/ \
--facesfolder data/facecache/ffpp/ \
--videodf data/ffpp_videos.pkl \
--facesdf data/ffpp_faces.pkl \
--checkpoint tmp/ffpp_prep/