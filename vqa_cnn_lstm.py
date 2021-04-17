import comet_ml

import torch

from train import train
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models, transforms
from config import *

from torch.utils.data import DataLoader
from pretrained import initialize_model
from datetime import datetime

from utils import weights_init

from params.vqa_cnn_lstm import *
from params.scheduler import *

from vqa import VQA
from preprocessing.vocabulary import Vocabulary

from pretrained import set_parameter_requires_grad

from dataloader import get_dataloaders

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # TODO find mean/std for train/val coco
    ])

class OriginalVQA(pl.LightningModule):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering paper (Zhou et al, 2017).
    """
    def __init__(self, question_vocab_size, ans_vocab_size, word_embed_size=300, hidden_size=512, num_layers=2, embed_size=1024, feature_extract=True, input_size=224):
        super(OriginalVQA, self).__init__()
        
        # input size to CNN
        self.input_size = input_size

        # load pretrained cnn
        self.cnn = models.vgg19(pretrained=True)
        
        # get size of last hidden layer of pretrained CNN
        num_ftrs = self.cnn.classifier[-1].in_features

        # remove last layer from CNN
        self.cnn.classifier = torch.nn.Sequential(*list(self.cnn.classifier.children())[:-1])

        # freeze all layers if using CNN as feature extractor
        set_parameter_requires_grad(self.cnn, feature_extract)
        
        # linear layer for image channel
        self.fc_image = nn.Linear(num_ftrs,embed_size)      
                
        # lstm
        self.word2vec = nn.Embedding(question_vocab_size, word_embed_size, padding_idx=VQA.questions_vocabulary.word2idx(Vocabulary.PAD_TOKEN))
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers, batch_first=True)
        self.fc_questions = nn.Linear(2*num_layers*hidden_size, embed_size)

        # mlp
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

        # activation funcitons
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        
        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # initialize parameters for fc layers
        weights_init(self.lstm)
        weights_init(self.fc1) 
        weights_init(self.fc2) 
        weights_init(self.fc_questions)
        weights_init(self.fc_image)

        # save hyperparameters
        self.save_hyperparameters()

    def forward(self, image, question_indices):
        """ 
        Since we are using Pytorch Lightning, the forward method defines how the LightningModule behaves during inference/prediction. 
        """
        # getting visual features from last hidden layer of pretrained CNN
        img_features = self.cnn(image)

        # normalize output as per https://arxiv.org/pdf/1505.00468.pdf
        l2_norm = torch.linalg.norm(img_features, ord=2, dim=1).reshape(-1,1)
        img_features = torch.div(img_features, l2_norm)

        # fully connected layer for image 
        img_features = self.leaky_relu(self.fc_image(img_features))
        
        # each question word is encoded with (Linear, Dropout(0.5), Tanh)
        question_embeddings = self.tanh(self.word2vec(question_indices))
        
        # question embedding obtained from the LSTM is a concatenation of last cell state and last hidden state representations from each of the hidden layers
        _, (hidden, cell) = self.lstm(question_embeddings)
        
        # concatenate cell state and hidden state of last layer
        question_features = torch.cat((hidden, cell), dim=2)

        # transpose since output of the LSTM has batch on second dimension
        question_features = torch.transpose(question_features, 0, 1).reshape(question_embeddings.size(0), -1)
        question_features = self.leaky_relu(self.fc_questions(question_features))
    
        # point-wise multiplication
        combined_feature = self.dropout(self.leaky_relu(torch.mul(img_features, question_features)))

        # a fully connected neural network classifier with 2 hidden layers and 1000 hidden units (dropout 0.5) in each layer with tanh non-linearity
        combined_feature = self.fc1(combined_feature)       
        combined_feature = self.leaky_relu(combined_feature)
        combined_feature = self.dropout(combined_feature)
        logits = self.fc2(combined_feature)

        return logits

    def training_step(self, batch, batch_idx):
        """ 
        training_step method defines a single iteration in the training loop. 
        """
        image            = batch["image"]
        question_indices = batch["question"]
        answer           = batch["answer"]
        q_ids            = batch["question_id"]
        answers          = batch["answers"]
        
        # get predictions using forward method 
        preds = self(image, question_indices)
        
        # CrossEntropyLoss expects class indices and not one-hot encoded vector as the target
        _, labels = torch.max(answer, dim=1)
        
        # compute CE loss
        loss = self.criterion(preds, labels)
        
        # logging training loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # TODO compute train_acc using q_ids and answers similar to basic_vqa

        return loss

    def validation_step(self, batch, batch_idx):
        """ 
        validation_step method defines a single iteration in the validation loop. 
        """
        image            = batch["image"]
        question_indices = batch["question"]
        answer           = batch["answer"]
        q_ids            = batch["question_id"]
        answers          = batch["answers"]
        
        # get predictions using forward method 
        preds = self(image, question_indices)
        
        # CrossEntropyLoss expects class indices and not one-hot encoded vector as the target
        _, labels = torch.max(answer, dim=1)

        # compute CE loss
        loss = self.criterion(preds, labels)
        
        # logging validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # TODO compute val_acc using q_ids

        return loss

    def configure_optimizers(self):
        """ 
        Configure our optimizers.
        """
        # creating optimizer for our model
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=verbose, factor=factor, patience=patience)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitored_loss}

if __name__ == "__main__":
    dataloaders = get_dataloaders(preprocess, batch_size,shuffle, num_workers)

    model = OriginalVQA(
        question_vocab_size=VQA.questions_vocabulary.size,
        ans_vocab_size=VQA.answers_vocabulary.size
        )

    train(model, dataloaders["train"], dataloaders["val"], epochs)
