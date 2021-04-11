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
from config import *

from train import train

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from vqa import VQA
from preprocessing.vocabulary import Vocabulary

# setting the seed for reproducibility (it is important to set seed when using DPP mode)
seed_everything(7)


class SimpleBaselineVQA(pl.LightningModule):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering paper (Zhou et al, 2017).
    """
    def __init__(self, questions_vocab_size, answers_vocab_size=1000, hidden_size=1024, word_embeddings_size=300):
        super(SimpleBaselineVQA, self).__init__()
        
        # the output size of Imagenet is 1000 and we want to resize it to 1024
        self.googlenet, self.input_size = initialize_model("googlenet", hidden_size, feature_extract=True, use_pretrained=True)
        self.embed_questions = nn.Embedding(questions_vocab_size, word_embeddings_size, padding_idx=VQA.questions_vocabulary.word2idx(Vocabulary.PAD_TOKEN))
        self.fc = nn.Linear(word_embeddings_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, answers_vocab_size)
        print("Multiplying Image and Text channels")
        # using negative log likelihood as loss
        self.criterion = nn.CrossEntropyLoss()
        
        # activation
        self.leaky_relu = nn.LeakyReLU()

        # initialize parameters
        weights_init(self.embed_questions)
        weights_init(self.fc)
        weights_init(self.fc2)

        # save hyperparameters
        self.save_hyperparameters()

    def forward(self, image, question_indices):
        """ 
        Since we are using Pytorch Lightning, the forward method defines how the LightningModule behaves during inference/prediction. 
        """
        # getting visual features
        img_feat = self.leaky_relu(self.googlenet(image))

        # getting language features
        word_embeddings = self.embed_questions(question_indices)

        # get embedding for question using average  # TODO: try sum vs average of word embeddings 
        ques_features = torch.mean(word_embeddings, dim=1)
        
        # fully connected layer taking word embeddings
        ques_features = self.leaky_relu(self.fc(ques_features))

        # concatenate features TODO: investigate concatenation vs element-wise multiplication
        features = torch.mul(img_feat, ques_features)

        # one fully connected layer
        logits = self.fc2(features)

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

        return optimizer

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
