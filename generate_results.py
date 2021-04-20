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
    paths = ["models/SimpleBaselineVQA/Apr-17-02-20-49/epoch=7-step=110943.ckpt",
                "models/OriginalVQA/Apr-17-02-20-49/epoch=8-step=124811.ckpt",
                "models/SAN/Apr-16-02-46-21/epoch=35-step=499247.ckpt ",
                "models/SAN/Apr-17-05-17-59/epoch=5-step=83207.ckpt",
                "models/SAN/Apr-17-21-32-59/epoch=3-step=55471.ckpt ",
                "models/SimpleBaselineVQA/Apr-17-21-31-10/epoch=3-step=55471.ckpt",
                "models/SimpleBaselineVQA/Apr-17-21-31-10/epoch=2-step=41603.ckpt",
                "models/SimpleBaselineVQA/Apr-17-21-31-10/last.ckpt",
                "models/OriginalVQA/Apr-17-21-31-15/epoch=4-step=69339.ckpt",
                "models/OriginalVQA/Apr-17-21-31-15/epoch=5-step=83207.ckpt",
                "models/OriginalVQA/Apr-17-21-31-15/last.ckpt",
                "models/SimpleBaselineVQA/Apr-17-21-37-49/epoch=4-step=69339.ckpt",
                "models/SimpleBaselineVQA/Apr-17-21-37-49/epoch=5-step=83207.ckpt",
                "models/SimpleBaselineVQA/Apr-17-21-37-49/last.ckpt",
                "models/OriginalVQA/Apr-17-21-37-48/epoch=4-step=69339.ckpt ",
                "models/OriginalVQA/Apr-17-21-37-48/epoch=6-step=97075.ckpt",
                "models/OriginalVQA/Apr-17-21-37-48/last.ckpt",
                "models/SimpleBaselineVQA/Apr-17-21-42-53/epoch=3-step=55471.ckpt  ",
                "models/SimpleBaselineVQA/Apr-17-21-42-53/last.ckpt",
                "models/OriginalVQA/Apr-17-21-42-53/epoch=4-step=69339.ckpt  ",
                "models/OriginalVQA/Apr-17-21-42-53/last.ckpt",
                "models/SimpleBaselineVQA/Apr-17-23-08-12/epoch=14-step=308489.ckpt",
                "models/OriginalVQA/Apr-17-23-08-11/epoch=14-step=308489.ckpt",
                "models/OriginalVQA/Apr-17-23-03-24/epoch=6-step=97075.ckpt",
                "models/OriginalVQA/Apr-17-23-03-24/epoch=8-step=124811.ckpt",
                "models/OriginalVQA/Apr-17-23-03-24/last.ckpt",
                "models/SimpleBaselineVQA/Apr-18-16-34-58/last.ckpt",
                "models/OriginalVQA/Apr-18-16-34-56/last.ckpt",
                "models/SAN/Apr-17-21-32-59/last.ckpt",
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