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

from params.trainer import *
from params.comet import *

import logging

file_logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_dataloader: DataLoader, epochs: int, val_dataloader:DataLoader = None):
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

    if val_dataloader:
      # early stoppping
      early_stopping_callback = EarlyStopping(
        monitor='val_loss', # monitor validation loss
        verbose=True, # log early-stop events
        patience=patience,
        min_delta=0.00, # minimum change is 0
        mode="min"
        )

      # update checkpoints based on validation loss by using ModelCheckpoint callback monitoring 'val_loss'
      checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                            mode="min",
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

      trainer.fit(model=model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
    else:
      checkpoint_callback = ModelCheckpoint(dirpath=folder)

      # define trainer 
      trainer = pl.Trainer(
        default_root_dir=folder, # Lightning automates saving and loading checkpoints
        max_epochs=epochs, 
        gpus=-1,
        logger=logger, 
        progress_bar_refresh_rate=30,
        callbacks=[checkpoint_callback])
        )

      trainer.fit(model=model, train_dataloader=train_dataloader)