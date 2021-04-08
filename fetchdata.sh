#! /bin/bash

# create a datasets directory
mkdir datasets
cd datasets

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
mkdir Images/mscoco

# unzip datasets
unzip train2014.zip  -d Images/mscoco
unzip val2014.zip -d Images/mscoco
unzip test2015.zip -d Images/mscoco
unzip v2_Annotations_Train_mscoco.zip -d Annotations
unzip v2_Annotations_Val_mscoco.zip -d Annotations
unzip v2_Complementary_Pairs_Train_mscoco.zip -d "Complementary Pairs"
unzip v2_Complementary_Pairs_Val_mscoco.zip -d "Complementary Pairs"
unzip v2_Questions_Train_mscoco.zip -d Questions
unzip v2_Questions_Val_mscoco.zip -d Questions
unzip v2_Questions_Test_mscoco.zip -d Questions

# delete all zip files
rm *.zip