from config import *
from vqa import VQA, VQATest

from torchvision import models, transforms

from torch.utils.data import DataLoader

# initialize training and validation dataset
def get_dataloaders(preprocess: transforms.Compose, batch_size, shuffle, num_workers, train=True, val=True, test=False, final_train=False):
    """ Return the dataloaders depending on whether we are training or testing """
    dataloaders = {}

    if not train and not val and not test:
        return dataloaders
    
    if train and final_train:
        train_dataset = VQA(
            dataset_file=preprocessed_train_val,
            transform=preprocess
            ) 

        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            collate_fn=VQA.vqa_collate
        )
    elif train:
        train_dataset = VQA(
            dataset_file=preprocessed_train,
            transform=preprocess
            ) 
        
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            collate_fn=VQA.vqa_collate
        )
    
    if val and not final_train: # no validation on final train

        val_dataset = VQA(
            dataset_file=preprocessed_val,
            transform=preprocess
        )
        
        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            collate_fn=VQA.vqa_collate
        )

    if test:    
        test_dataset = VQATest(
            dataset_file=preprocessed_test,
            transform=preprocess
        )
              
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
    
    return dataloaders