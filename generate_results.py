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

if __name__ == '__main__':
    MODEL_PATH = 'models/OriginalVQA/4a618c13c47a4fd3955355bfd9546c41/checkpoints/epoch=2-step=41603.ckpt'

    # wether evaluation on test-dev or test-standard
    test_dev = False
    results = []

    # create test dataset
    test_dataset = VQATest(
        testdev_quesFile if test_dev else test_quesFile,
        test_imgDir,
        transform=original_preprocess
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # load trained model
    model = OriginalVQA.load_from_checkpoint(checkpoint_path=MODEL_PATH)
    
    # determine dataSubType
    dataSubType = testdev_dataSubType if test_dev else test_dataSubType
    resultFile = os.path.join("Results", f"{type(model).__name__}_{taskType}_{dataType}_{dataSubType}_results.json")

    # freeze all layers of the model and set it to evaluation mode
    model.freeze()
    total_batches = len(test_loader)
    for idx, batch in enumerate(test_loader):
        if idx % 50:
            print(f"Done batch number {idx} out of {total_batches}")
        output = model(batch["image"], batch["question"])
        _, preds = torch.max(output, dim=1)
        for pred, q_id in zip(preds, batch["question_id"].tolist()):
            answer = VQA.answers_vocabulary.idx2word(pred)
            results.append({"question_id": q_id , "answer": answer})

    with open(resultFile, 'w') as outfile:
        json.dump(results, outfile)
