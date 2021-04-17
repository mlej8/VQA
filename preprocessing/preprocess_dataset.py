import json
import os
import argparse
import datetime

from config import *

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

img_path ='COCO_%s_%012d.jpg'

def preprocess_train_val():
    train = []
    val = []
    logger.info('Loading VQA training and validation annotations and questions into memory...')
    time_t = datetime.datetime.utcnow()
    train_ann = json.load(open(train_annFile, 'r'))
    val_ann = json.load(open(val_annFile, 'r'))

    train_ques = json.load(open(train_quesFile, 'r'))
    val_ques = json.load(open(val_quesFile, 'r'))
    logger.info("Done in {}".format(datetime.datetime.utcnow() - time_t))

    # creating dictionaries mapping question ids to question object
    logger.info('Creating indexes for questions...')
    train_ques_id = {}
    val_ques_id = {}
    for ques in train_ques['questions']:
            train_ques_id[ques['question_id']] = ques
    for ques in val_ques['questions']:
            val_ques_id[ques['question_id']] = ques
    logger.info('Index created!')
    
    logger.info('Creating training dataset.')
    for ann in train_ann['annotations']:
        sample = {
            'question_type': ann['question_type'], 
            'multiple_choice_answer': ann['multiple_choice_answer'], 
            'answers': ann['answers'], 
            'image_id': ann['image_id'], 
            'answer_type': ann['answer_type'], 
            'question_id': ann['question_id'],
            'question': train_ques_id[ann['question_id']]["question"], 
            'image_path': os.path.join(train_imgDir, img_path % (train_dataSubType, ann['image_id']))
        }

        train.append(sample)
    logger.info('Created training dataset.')

    logger.info('Creating validation dataset.')
    for ann in val_ann['annotations']:
        sample = {
            'question_type': ann['question_type'], 
            'multiple_choice_answer': ann['multiple_choice_answer'], 
            'answers': ann['answers'], 
            'image_id': ann['image_id'], 
            'answer_type': ann['answer_type'], 
            'question_id': ann['question_id'],
            'question': val_ques_id[ann['question_id']]["question"], 
            'image_path': os.path.join(val_imgDir, img_path % (val_dataSubType, ann['image_id']))
        }

        val.append(sample)
    logger.info('Created validation dataset!')
    
    # concatenate train and val for final evaluation
    train_val = train + val
    logger.info(f'Training dataset length: {len(train)}')
    logger.info(f'Validation dataset length: {len(val)}')
    logger.info(f'Train + val dataset length: {len(train_val)}')

    # write to json files
    json.dump(train, open(preprocessed_train, 'w'))
    logger.info(f"Done writing training dataset to {preprocessed_train}")
    json.dump(val, open(preprocessed_val, 'w'))
    logger.info(f"Done writing validation dataset to {preprocessed_val}")
    json.dump(train_val, open(preprocessed_train_val, 'w'))
    logger.info(f"Done writing training + validation dataset to {preprocessed_train_val}")

def preprocess_test():
    test = []
    test_dev = []

    test_ques = json.load(open(test_quesFile, 'r'))
    testdev_ques = json.load(open(testdev_quesFile, 'r'))
    logger.info('Creating test-standard dataset.')
    for ques in test_ques['questions']:
        sample = {
            'image_id': ques['image_id'], 
            'question_id': ques['question_id'],
            'question': ques["question"], 
            'image_path': os.path.join(test_imgDir, img_path % (test_dataSubType, ques['image_id']))
        }

        test.append(sample)
    logger.info('Created test-standard dataset!')

    logger.info('Creating test-dev dataset.')
    for ques in testdev_ques['questions']:
        sample = {
            'image_id': ques['image_id'], 
            'question_id': ques['question_id'],
            'question': ques["question"], 
            'image_path': os.path.join(test_imgDir, img_path % (test_dataSubType, ques['image_id']))
        }

        test_dev.append(sample)
    logger.info('Created test-dev dataset!')
    
    # concatenate datasets
    test_complete = test + test_dev
    logger.info(f'Test-standard dataset length: {len(test)}')
    logger.info(f'Test-dev dataset length: {len(test_dev)}')
    logger.info(f'Test (dev + standard) dataset length: {len(test_complete)}')
    logger.info('Created all datasets.')


    # write datasets
    json.dump(test, open(preprocessed_test_standard, 'w'))
    logger.info(f"Done writing test-standard dataset to {preprocessed_test_standard}")
    json.dump(test_dev, open(preprocessed_test_dev, 'w'))
    logger.info(f"Done writing test-dev dataset to {preprocessed_test_dev}")
    json.dump(test_complete, open(preprocessed_test, 'w'))
    logger.info(f"Done writing test (dev + standard) dataset to {preprocessed_test}")

if __name__ == "__main__":
    """ 
    This script preprocess VQA datasets to create the following five datasets:
        1. train 
        2. val
        3. train + val
        4. test-standard
        5. test-dev
        6. test-standard + test-dev

    The first three datasets will have the following format:
    [
        {
            'question_type', 
            'multiple_choice_answer', 
            'answers', 
            'image_id', 
            'answer_type', 
            'question_id',
            'question', 
            'image_path'
        },
    ...]

    The last three (test datasets) will have the following format:
    [
        {
            'image_id' 
            'question_id',
            'question' 
            'image_path'
        }
    ...]
    """
    preprocess_train_val()
    preprocess_test()