#! /bin/bash

# questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip

# annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

# complementary pair list
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip

# VQA input images
wget http://images.cocodataset.org/zips/train2014.zip http://images.cocodataset.org/zips/val2014.zip http://images.cocodataset.org/zips/test2015.zip

# create directories for data
mkdir Annotations
mkdir "Complementary Pairs"
mkdir Questions
mkdir Images

# unzip images
unzip train2014.zip val2014.zip test2015.zip -d Images
unzip v2_Annotations_Train_mscoco.zip v2_Annotations_Val_mscoco.zip -d Annotations
unzip v2_Complementary_Pairs_Train_mscoco.zip v2_Complementary_Pairs_Val_mscoco.zip -d "Complementary Pairs"
unzip v2_Questions_Train_mscoco.zip v2_Questions_Val_mscoco.zip v2_Questions_Test_mscoco.zip -d Questions