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

# test VQA dataset
from vqa import VQA, VQATest
import logging
logger = logging.getLogger(__name__)
if __name__ == '__main__':
    OR_MODEL_PATH = "models/OriginalVQA/Apr-17-02-20-49/OriginalVQA/56972bc21c92492b94fd8f0252d4b2f2/checkpoints/epoch=8-step=124811.ckpt"
    SB_MODEL_PATH = "models/SimpleBaselineVQA/Apr-17-02-20-49/SimpleBaselineVQA/6155dbd7b8354c409344924ce3ce16ac/checkpoints/epoch=7-step=110943.ckpt"
    SA_MODEL_PATH = "models/SAN/"

    # wether evaluation on test-dev or test-standard
    test_dev = False
    results = []
    """ 
    
    """
    # create test dataset
    testdev_dataset = VQATest(
        testdev_quesFile,
        test_imgDir,
        transform=original_preprocess
    )

    test_dataset = VQATest(
        test_quesFile,
        test_imgDir,
        transform=original_preprocess
    )
    
    testdev_loader = DataLoader(
        testdev_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # load trained model
    model = OriginalVQA.load_from_checkpoint(checkpoint_path=OR_MODEL_PATH, question_vocab_size=VQA.questions_vocabulary.size, ans_vocab_size=VQA.answers_vocabulary.size)
    # model = SimpleBaselineVQA.load_from_checkpoint(checkpoint_path=SB_MODEL_PATH, in_dim=VQA.questions_vocabulary.size, out_dim=VQA.answers_vocabulary.size)
    # model = SAN.load_from_checkpoint(checkpoint_path=SA_MODEL_PATH, in_dim=VQA.questions_vocabulary.size, out_dim=VQA.answers_vocabulary.size)

    # determine dataSubType
    resultFile = os.path.join("Results", f"{type(model).__name__}_{taskType}_{dataType}_results.json")

    # freeze all layers of the model and set it to evaluation mode
    model.freeze()

    logger.info(f"Predicting on test standard")
    for idx, batch in enumerate(test_loader):
        logger.info(f"Done batch {idx}")
        output = model(batch["image"], batch["question"])
        _, preds = torch.max(output, dim=1)
        for pred, q_id in zip(preds, batch["question_id"].tolist()):
            answer = VQA.answers_vocabulary.idx2word(pred)
            results.append({"question_id": q_id , "answer": answer})
    logger.info(f"Done on test standard")
    logger.info(f"Predicting on test dev")
    for idx, batch in enumerate(testdev_loader):
        logger.info(f"Done batch {idx}")
        output = model(batch["image"], batch["question"])
        _, preds = torch.max(output, dim=1)
        for pred, q_id in zip(preds, batch["question_id"].tolist()):
            answer = VQA.answers_vocabulary.idx2word(pred)
            results.append({"question_id": q_id , "answer": answer})
    logger.info(f"Done on test dev")
    
    with open(resultFile, 'w') as outfile:
        json.dump(results, outfile)
