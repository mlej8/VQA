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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_dataloader: DataLoader, val_dataloader:DataLoader, epochs: int, patience:int = 5):
    # create folder for each run
    folder = "models/{}".format(datetime.now().strftime("%b-%d-%H-%M-%S"))
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

    logger = loggers.CometLogger( 
            save_dir="logs/",
            workspace="vqa",
            project_name=f"{type(model).__name__}", 
            experiment_name=f"{type(model).__name__}_{datetime.now().strftime('%b_%d_%H_%M_%S')}"
        )

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
    PATH = folder + '/result'
    with open(PATH, "w") as f:
        f.write(f"Model: {str(model)}\n")
        f.write(f"Lowest training loss: {str(min(logger.get_metric('train_loss')))}\n")
        f.write(f"Lowest validation loss: {str(min(logger.get_metric('val_loss')))}\n")

    # plot training vs validation loss
    plt.plot(range(len(logger.get_metric('train_loss'))), logger.get_metric('train_loss'), lw=2, label='Training Loss')
    plt.plot(range(len(logger.get_metric('val_loss'))), logger.get_metric('val_loss'), lw=2, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(folder + f"/{type(model).__name__}_training_validation_loss.png")
    plt.show()
