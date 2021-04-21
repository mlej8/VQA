import json
import os

from config import *

# models
from simple_vqa_baseline import SimpleBaselineVQA
from vqa_cnn_lstm import OriginalVQA
from san import SAN

# pytorch
import torch
from torch.utils.data import DataLoader

# preprocess method
from simple_vqa_baseline import preprocess as simple_preprocess
from vqa_cnn_lstm import preprocess as original_preprocess
from san import preprocess as san_preprocess

from dataloader import get_dataloaders
from train import test

# test VQA dataset
import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # NOTE: include path to checkpoint of models for which we want to generate results
    paths = [
        "models/SimpleSAN/Apr-20-22-56-29/last.ckpt"
    ]

    for PATH in reversed(paths): 
        PATH = PATH.strip()
        model = PATH.split("/")[1]
        if model == "SimpleBaselineVQA":
            result_path = test(PATH=PATH,model_class=SimpleBaselineVQA,preprocess=simple_preprocess)
        elif model == "SAN":
            result_path = test(PATH=PATH,model_class=SAN,preprocess=san_preprocess)
        elif model == "OriginalVQA":
            result_path = test(PATH=PATH,model_class=OriginalVQA,preprocess=original_preprocess)
        logger.info(f"Done generating results for {PATH} {model} at {result_path}")
