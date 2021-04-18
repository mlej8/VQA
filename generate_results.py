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

# test VQA dataset
from vqa import VQA, VQATest
import logging
logger = logging.getLogger(__name__)
if __name__ == '__main__':
    # model path # NOTE: change this for each model
    OR_MODEL_PATH = "models/OriginalVQA/Apr-17-02-20-49/OriginalVQA/56972bc21c92492b94fd8f0252d4b2f2/checkpoints/epoch=8-step=124811.ckpt"
    SB_MODEL_PATH = "models/SimpleBaselineVQA/Apr-17-02-20-49/SimpleBaselineVQA/6155dbd7b8354c409344924ce3ce16ac/checkpoints/epoch=7-step=110943.ckpt"
    SA_MODEL_PATH = "models/SAN/"

    results = []
    
    # preprocessing (transformation)
    preprocess = original_preprocess # NOTE: change this for each model

    # load trained model NOTE: change this for each model
    model = OriginalVQA.load_from_checkpoint(checkpoint_path=OR_MODEL_PATH, question_vocab_size=VQA.questions_vocabulary.size, ans_vocab_size=VQA.answers_vocabulary.size)
    # model = SimpleBaselineVQA.load_from_checkpoint(checkpoint_path=SB_MODEL_PATH, question_vocab_size=VQA.questions_vocabulary.size, ans_vocab_size=VQA.answers_vocabulary.size)
    # model = SAN.load_from_checkpoint(checkpoint_path=SA_MODEL_PATH, question_vocab_size=VQA.questions_vocabulary.size, ans_vocab_size=VQA.answers_vocabulary.size)

    # freeze all layers of the model and set it to evaluation mode
    model.freeze()

    # get test loader
    test_loader = get_dataloaders(preprocess, 32, False, 8, train=False, val=False, test=True)["test"]

    # generate result file name
    resultFile = os.path.join("Results", f"{type(model).__name__}_{versionType}{taskType}_{dataType}_results.json")
    
    logger.info(f"Predicting on test dataset (standard + dev)")
    total_batch_num = len(test_loader)
    for idx, batch in enumerate(test_loader):
        output = model(batch["image"], batch["question"])
        _, preds = torch.max(output, dim=1)
        results += [{
            "answer": VQA.answers_vocabulary.idx2word(pred),
            "question_id": q_id
            } 
            for pred, q_id in zip(preds.tolist(), batch["question_id"].tolist())]
        logger.info(f"Done batch {idx} out of {total_batch_num}")
        
    logger.info(f"Done prediction!")
    
    with open(resultFile, 'w') as outfile:
        json.dump(results, outfile)
