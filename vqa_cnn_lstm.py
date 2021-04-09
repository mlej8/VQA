import comet_ml
import torch

from train import train
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
from config import *
from pytorch_lightning.utilities.seed import seed_everything

from torch.utils.data import DataLoader
from pretrained import initialize_model
from datetime import datetime

from utils import weights_init

from params.vqa_cnn_lstm import *

from vqa import VQA, vqa_collate

# setting the seed for reproducability (it is important to set seed when using DPP mode)
seed_everything(7)

class OriginalVQA(pl.LightningModule):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering paper (Zhou et al, 2017).
    """
    def __init__(self, question_vocab_size, ans_vocab_size=1000, word_embed_size=300, hidden_size=512, num_layers=2, embed_size=2048):
        super(OriginalVQA, self).__init__()
        
        # the output size of Imagenet is 1000 and we want to resize it to 1024
        self.cnn, self.input_size = initialize_model("vgg19", 1024, True, use_pretrained=True) # TODO question: are we feature extracting or finetuning ? we can try both.. we are doing feature extract for the moment (e.g. not updating params of pretrained network) but might be useful to try finetuning should get better results
        
        # lstm
        self.word2vec = nn.Embedding(question_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc_questions = nn.Linear(2*num_layers*hidden_size, embed_size)

        # vqa model
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)
        self.softmax = nn.Softmax()
        self.leaky_relu = nn.LeakyReLU()

        self.dropout = nn.Dropout(0.2)
        
        self.criterion = nn.CrossEntropyLoss()

        # initialize parameters for fc layers
        weights_init(self.lstm) # TODO add check to initialize the LSTM layer in utils.py
        weights_init(self.fc1) 
        weights_init(self.fc2) 
        weights_init(self.fc_questions)

    def forward(self, image, question):
        """ 
        Since we are using Pytorch Lightning, the forward method defines how the LightningModule behaves during inference/prediction. 
        """
        # getting visual features
        img_features = self.cnn(image)

        # TODO maybe normalize the output ? see code below
        # l2_norm = torch.linalg.norm(img_features, p=2, dim=1)
        # img_feature = torch.div(img_feature, l2_norm)
        
        # lstm
        question_vector = self.word2vec(question.int())                             
        question_vector = self.tanh(question_vector)
        question_vector = question_vector.transpose(0, 1)                             
        _, (hidden, cell) = self.lstm(question_vector)                        
        question_features = torch.cat((hidden, cell), 2)                    
        question_features = question_features.transpose(0, 1)                     
        question_features = question_features.reshape(question_features.size()[0], -1)  
        question_features = self.tanh(question_features)
        question_features = self.fc_questions(question_features)                            
    
        # concatenate features        
        combined_features = torch.cat((img_features, question_features), 1)
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)       
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        logits = self.fc2(combined_feature)

        return self.softmax(logits)

    def training_step(self, batch, batch_idx):
        """ 
        training_step method defines a single iteration in the training loop. 
        """

        # The LightningModule knows what device it is on - you can reference via `self.device`, it makes your models hardware agnostic (you can train on any number of GPUs spread out on differnet machines)
        (image, question_encodings, answers) = batch
        
        # get predictions using forward method 
        preds = self(image, question_encodings)
        
        # CrossEntropyLoss expects class indices and not one-hot encoded vector as the target
        _, labels = torch.max(answers, dim=1)
        
        # compute CE loss
        loss = self.criterion(preds, labels)
        
        # logging training loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """ 
        validation_step method defines a single iteration in the validation loop. 
        """

        # The LightningModule knows what device it is on - you can reference via `self.device`
        (image, question_encodings, answers) = batch
        
        # get predictions using forward method 
        preds = self(image, question_encodings)
        
        # CrossEntropyLoss expects class indices and not one-hot encoded vector as the target
        _, labels = torch.max(answers, dim=1)

        # compute CE loss
        loss = self.criterion(preds, labels)
        
        # logging validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    def configure_optimizers(self):
        """ 
        Configure our optimizers.
        """

        # creating optimizer for our model
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        return optimizer

if __name__ == "__main__":
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # TODO find mean/std for train/val coco
        ])
    
    # initialize training and validation dataset
    train_dataset = VQA(
        train_annFile,
        train_quesFile,
        train_imgDir,
        transform=preprocess
        ) 
    val_dataset = VQA(
        val_annFile,
        val_quesFile,
        val_imgDir,
        transform=preprocess
        ) 
    
    train_dataloader = DataLoader(
    	train_dataset,
    	batch_size=batch_size, 
    	shuffle=shuffle, 
    	num_workers=num_workers,
        collate_fn=vqa_collate
	)
    
    val_dataloader = DataLoader(
    	val_dataset,
    	batch_size=batch_size, 
    	shuffle=False,  # set False for validation dataloader
    	num_workers=num_workers,
        collate_fn=vqa_collate
	)

    model = OriginalVQA(
        question_vocab_size=VQA.questions_vocabulary.size,
        ans_vocab_size=VQA.answers_vocabulary.size
        )

    train(model, train_dataloader, val_dataloader, epochs)