import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning import loggers 

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from params.trainer import *
from params.comet import *
from params.test import *

from dataloader import get_dataloaders

from config import *

from vqa import VQA

import logging

file_logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
            model, 
            preprocess: transforms.Compose, 
            batch_size: int,  
            shuffle: int,  
            num_workers: int,  
            final_train: bool, 
            epochs: int, 
            resume_ckpt: str = None):
    if final_train:
        dataloaders = get_dataloaders(preprocess, batch_size, shuffle, num_workers, final_train=final_train)
    else:
        dataloaders = get_dataloaders(preprocess, batch_size, shuffle, num_workers)
    
    # create folder for each run
    folder = "models/{}/{}".format(type(model).__name__, datetime.now().strftime("%b-%d-%H-%M-%S"))
    os.makedirs(folder, exist_ok=True)

    # log model architecture
    file_logger.info(f"Model: {str(model)}")
    file_logger.info(f"Logs are stored at: {folder}")

    # initializing loggers
    logger = loggers.CometLogger( 
            save_dir=folder,
            workspace=workspace,
            project_name=project_name, 
            experiment_name=f"{type(model).__name__}_{datetime.now().strftime('%b_%d_%H_%M_%S')}"
        )

    if resume_ckpt:
      # resume training from checkpoint
      trainer = pl.Trainer(resume_from_checkpoint=resume_ckpt)

      # automatically restores model, epoch, step, LR schedulers, apex, etc...
      trainer.fit(model)
    elif dataloaders.get("val") is not None:
      # early stoppping
      early_stopping_callback = EarlyStopping(
        monitor='val_acc', # monitor validation accuracy
        verbose=True, # log early-stop events
        patience=patience,
        min_delta=0.00, # minimum change is 0
        mode="max"
        )

      # update checkpoints based on validation loss by using ModelCheckpoint callback monitoring 'val_loss'
      checkpoint_callback = ModelCheckpoint(monitor='val_acc',
                                            mode="max",
                                          save_top_k=top_k,
                                          save_last=True,
                                          dirpath=folder)

      # define trainer 
      trainer = pl.Trainer(
        default_root_dir=folder, # Lightning automates saving and loading checkpoints
        max_epochs=epochs, 
        gpus=-1,
        logger=logger, 
        progress_bar_refresh_rate=30, 
        callbacks=[early_stopping_callback, checkpoint_callback])

      trainer.fit(model=model, train_dataloader=dataloaders["train"], val_dataloaders=dataloaders["val"])
    else:
      # final training (train + val)
      checkpoint_callback = ModelCheckpoint(dirpath=folder)

      # define trainer 
      trainer = pl.Trainer(
        default_root_dir=folder, # Lightning automates saving and loading checkpoints
        max_epochs=epochs, 
        gpus=-1,
        logger=logger, 
        progress_bar_refresh_rate=30,
        callbacks=[checkpoint_callback])

      trainer.fit(model=model, train_dataloader=dataloaders["train"])
    
    best_checkpoint_path = trainer.checkpoint_callback.best_model_path

    file_logger.info(f"Best checkpoint path {best_checkpoint_path} for {type(model).__name__}")
    
    test(PATH=best_checkpoint_path, model=model, preprocess=preprocess)

def test(PATH, model, preprocess):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_logger.info(f"Loading checkpoint at {PATH} for {type(model).__name__}")

    # load best model from checkpoint path
    model = model.__class__.load_from_checkpoint(
      checkpoint_path=PATH, 
      questions_vocab_size=VQA.questions_vocabulary.size, 
      answers_vocab_size=VQA.answers_vocabulary.size,  
      map_location=device
      )

    # freeze all layers of the model and set it to evaluation mode
    model.freeze()

    # get test loader
    test_loader = get_dataloaders(preprocess, test_batch_size, test_shuffle, test_num_workers, train=False, val=False, test=True)["test"]

    # generate result file name
    result_file = os.path.join("Results", f"{type(model).__name__}_{versionType}{taskType}_{dataType}_results_{PATH.split('/')[2]}.json")
    
    # list to store predictions
    results = []
    total_batch_num = len(test_loader)
    file_logger.info(f"Predicting on test dataset (standard + dev)")

    for idx, batch in enumerate(test_loader):
        image = batch["image"].to(device)
        question = batch["question"].to(device)
        output = model(image, question)
        _, preds = torch.max(output, dim=1)
        results += [{
            "answer": VQA.answers_vocabulary.idx2word(pred),
            "question_id": q_id
            } 
            for pred, q_id in zip(preds.cpu().tolist(), batch["question_id"].tolist())]
        file_logger.info(f"Done batch {idx} out of {total_batch_num}")
        
    file_logger.info(f"Done prediction!")

    with open(result_file, 'w') as outfile:
        json.dump(results, outfile)

    file_logger.info(f"Predictions stored at {result_file}!")