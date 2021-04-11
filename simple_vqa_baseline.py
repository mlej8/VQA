import comet_ml

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning.utilities.seed import seed_everything

from pretrained import initialize_model
from datetime import datetime

from utils import weights_init

from params.simple_vqa_baseline import *
from params.scheduler import *
from config import *

from train import train

from torch.utils.data import DataLoader
from torchvision import models, transforms

from vqa import VQA
from preprocessing.vocabulary import Vocabulary

from pretrained import set_parameter_requires_grad

# setting the seed for reproducibility (it is important to set seed when using DPP mode)
seed_everything(7)

class SimpleBaselineVQA(pl.LightningModule):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering paper (Zhou et al, 2017).
    """
    def __init__(self, questions_vocab_size, answers_vocab_size=1000, hidden_size=1024, word_embeddings_size=300, feature_extract=True, input_size=224):
        super(SimpleBaselineVQA, self).__init__()
        
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
        self.fc_image = nn.Linear(num_ftrs, hidden_size)  
        
        # question channel
        self.word2vec = nn.Embedding(questions_vocab_size, word_embeddings_size, padding_idx=VQA.questions_vocabulary.word2idx(Vocabulary.PAD_TOKEN))
        self.fc_questions = nn.Linear(word_embeddings_size, hidden_size)
        self.fc1 = nn.Linear(2*hidden_size, answers_vocab_size)
        self.fc2 = nn.Linear(answers_vocab_size, answers_vocab_size)
        
        # using negative log likelihood as loss
        self.criterion = nn.CrossEntropyLoss()
        
        # activation
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

        # initialize parameters
        weights_init(self.word2vec)
        weights_init(self.fc_image)
        weights_init(self.fc_questions)
        weights_init(self.fc1)
        weights_init(self.fc2)

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

        # getting word embeddings
        word_embeddings = self.tanh(self.word2vec(question_indices))

        # get embedding for question using average
        ques_features = torch.mean(word_embeddings, dim=1)
        
        # fully connected layer taking word embeddings
        ques_features = self.leaky_relu(self.fc_questions(ques_features))

        # concatenate features TODO: investigate concatenation vs element-wise multiplication
        combined_features = self.dropout(self.leaky_relu(torch.cat((img_features, ques_features), dim=1)))

        # one fully connected layer
        combined_features = self.fc1(combined_features)       
        combined_features = self.leaky_relu(combined_features)
        combined_features = self.dropout(combined_features)
        logits = self.fc2(combined_features)

        return logits

    def training_step(self, batch, batch_idx):
        """ 
        training_step method defines a single iteration in the training loop. 
        """

        # The LightningModule knows what device it is on - you can reference via `self.device`, it makes your models hardware agnostic (you can train on any number of GPUs spread out on differnet machines)
        (image, question_indices, answers) = batch

        # get predictions using forward method 
        preds = self(image, question_indices)

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
        (image, question_indices, answers) = batch

        # get predictions using forward method 
        preds = self(image, question_indices)

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=verbose, patience=patience, factor=factor)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitored_loss}

if __name__ == "__main__":
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # TODO find mean/std for train/val coco
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
        collate_fn=VQA.vqa_collate
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # set False for validation dataloader
        num_workers=num_workers,
        collate_fn=VQA.vqa_collate
    )

    model = SimpleBaselineVQA(
        questions_vocab_size=VQA.questions_vocabulary.size,
        answers_vocab_size=VQA.answers_vocabulary.size
    )

    train(model, train_dataloader, val_dataloader, epochs)
