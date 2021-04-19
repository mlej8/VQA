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

from dataloader import get_dataloaders
from train import test

# test VQA dataset
import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # model path # NOTE: change this for each model
    OR_MODEL_PATH = "models/OriginalVQA/Apr-18-16-34-56/epoch=4-step=69339.ckpt"
    SB_MODEL_PATH = "models/SimpleBaselineVQA/Apr-18-16-34-58/epoch=4-step=69339.ckpt"
    SA_MODEL_PATH = "models/SAN/"
    
    test(PATH=SB_MODEL_PATH,model_class=SimpleBaselineVQA,preprocess=simple_preprocess)