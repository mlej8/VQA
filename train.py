import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from dict_logger import DictLogger

import torch
from torch.utils.data import DataLoader

from params.trainer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_dataloader: DataLoader, val_dataloader:DataLoader, epochs: int):
    # create folder for each run
    folder = "models/{}/{}".format(type(model).__name__, datetime.now().strftime("%b-%d-%H-%M-%S"))
    if not os.path.exists(folder):
        os.makedirs(folder)

    # early stoppping
    early_stopping_callback = EarlyStopping(
      monitor='val_loss', # monitor validation loss
      verbose=True, # log early-stop events
      patience=patience,
      min_delta=0.00 # minimum change is 0
      )

    # update checkpoints based on validation loss by using ModelCheckpoint callback monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    logger = DictLogger()

    # define trainer 
    trainer = pl.Trainer(
      default_root_dir=folder, # Lightning automates saving and loading checkpoints
      max_epochs=epochs, gpus=-1,
      logger=logger, 
      progress_bar_refresh_rate=30, 
      callbacks=[early_stopping_callback, checkpoint_callback])

    # train
    trainer.fit(model=model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

    # save test result
    PATH = os.path.join(folder,'result')
    with open(PATH, "w") as f:
        f.write(f"Model: {str(model)}\n")
        f.write(json.dumps(logger.metrics))
        f.write("\n")
        f.write(f"Lowest training loss: {str(min(logger.metrics['train_loss']))}\n")
        f.write(f"Lowest validation loss: {str(min(logger.metrics['val_loss']))}\n")
        # f.write(f"Test loss: {result}")

    # plot training vs validation loss
    plt.plot(range(len(logger.get_metric('train_loss'))), logger.get_metric('train_loss'), lw=2, label='Training Loss')
    plt.plot(range(len(logger.get_metric('val_loss'))), logger.get_metric('val_loss'), lw=2, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.savefig(os.path.join(folder, f"{type(model).__name__}_training_validation_loss.png"))
    plt.show()