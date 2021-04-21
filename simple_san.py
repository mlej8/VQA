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

from params.san import *
from params.scheduler import *

from vqa import VQA
from preprocessing.vocabulary import Vocabulary

from pretrained import set_parameter_requires_grad

from dataloader import get_dataloaders

from san import preprocess as san_preprocess, Attention

class SimpleSAN(pl.LightningModule):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering paper (Zhou et al, 2017).
    """
    def __init__(self, questions_vocab_size, answers_vocab_size, word_embed_size=300, hidden_size=512, num_layers=2, embed_size=1024, feature_extract=True, num_attention_layer=2, dropout=0.5, final_train=False):
        super(SimpleSAN, self).__init__()

        # whether training with train + val (final training)
        self.final_train = final_train

        # only use the visual features from the last pooling layer since it retains spatial information of the original images
        self.cnn = models.vgg19(pretrained=True).features 

        # freeze all layers if using CNN as feature extractor
        set_parameter_requires_grad(self.cnn, feature_extract)
        
        # linear layer for image channel
        self.fc_image = nn.Linear(hidden_size,embed_size)      
                
        # lstm
        self.word2vec = nn.Embedding(questions_vocab_size, word_embed_size, padding_idx=VQA.questions_vocabulary.word2idx(Vocabulary.PAD_TOKEN))
        self.fc_questions = nn.Linear(word_embed_size, embed_size)

        # attention networks
        self.attention_layers = nn.ModuleList([Attention(embed_size, hidden_size)] * num_attention_layer)

        # classifier
        self.fc1 = nn.Linear(embed_size, answers_vocab_size)

        # activation funcitons
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        
        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # initialize parameters for fc layers
        weights_init(self.fc_image)
        weights_init(self.word2vec)
        weights_init(self.fc_questions)
        weights_init(self.fc1) 

        # save hyperparameters
        self.save_hyperparameters()

    def forward(self, image, question_indices):
        """ 
        Since we are using Pytorch Lightning, the forward method defines how the LightningModule behaves during inference/prediction. 
        """
        #####################
        ### IMAGE CHANNEL ###
        #####################
        # N * 3 * 448 * 448 -> N * 512 * 14 * 14 - 14 x 14 is the number of regions in the image and 512 is the dimension of the feature vector for each region
        img_features = self.cnn(image) # each pixel corresponds to a 32 x 32 pixel region of the original input image and we have 196 feature vectors for each image region

        # N * 512 * 14 * 14 -> N * 196 * 512
        img_features = img_features.reshape(img_features.size(0), -1, img_features.size(1))

        # N * 196 * 512 -> N * 196 * 1024 - use a single layer perceptron to transform each feature vector to a new vector that has the same dimension as the question vector
        img_features = self.leaky_relu(self.fc_image(img_features))
                
        ########################
        ### QUESTION CHANNEL ###
        ########################
        # getting embedding from embedding layer
        question_embeddings = self.tanh(self.word2vec(question_indices))
        
        # get embedding for question using average
        question_features = torch.mean(question_embeddings, dim=1)

        # N * 1024 -> N * 1024
        u = self.leaky_relu(self.fc_questions(question_features))

        ##########################
        ### ATTENTION NETWORKS ###
        ##########################
        for attention_layer in self.attention_layers:
            u = attention_layer(img_features, u)

        ##################
        ### CLASSIFIER ###
        ##################
        logits = self.fc1(u)       

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
        _, preds_idx = torch.max(preds,dim=1)
        predicted_words = [VQA.answers_vocabulary.idx2word(idx) for idx in preds_idx.tolist()]

        # compute CE loss
        loss = self.criterion(preds, labels)
        
        # compute the train_acc
        train_acc = sum([min(answers[i].count(word)/3, 1) for i, word in enumerate(predicted_words)]) / batch_size

        # logging training loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True, prog_bar=True)

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
        _, preds_idx = torch.max(preds, dim=1)
        predicted_words = [VQA.answers_vocabulary.idx2word(idx) for idx in preds_idx.tolist()]

        # compute CE loss
        loss = self.criterion(preds, labels)
        
        # compute the val_acc
        val_acc = sum([min(answers[i].count(word)/3, 1) for i, word in enumerate(predicted_words)]) / batch_size

        # logging validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """ 
        Configure our optimizers.
        """
        # creating optimizer for our model
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if self.final_train:
            return {"optimizer": optimizer}
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=verbose, patience=patience, factor=factor)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitored_loss}

if __name__ == "__main__":
    # final training or not
    final = False

    model = SimpleSAN(
        questions_vocab_size=VQA.questions_vocabulary.size,
        answers_vocab_size=VQA.answers_vocabulary.size,
        final_train=final
        )
        
    train(
        model=model,
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        preprocess=san_preprocess, 
        final_train=final, 
        epochs=opt_epochs if final else epochs
        )