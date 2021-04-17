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

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((448,448)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # TODO find mean/std for train/val coco
    ])

class Attention(torch.nn.Module):
    def __init__(self, d=1024, k=512, dropout=0.0):
        super(Attention, self).__init__()
        self.fc_questions = nn.Linear(d, k)
        self.fc_images = nn.Linear(d, k)

        if dropout:
            self.dropout = nn.Dropout(dropout)

        # single layer perceptron to learn the attention distribution
        self.attention_layer = nn.Linear(k, 1)
        
        # activations
        self.tanh = torch.nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # initialize parameters
        self.apply(weights_init)

    def forward(self, vi, vq):
        # N * 196 (m) * 1024 (d) -> N * 196 (m) * 512 (k)
        img_features = self.fc_images(vi)
                
        # N * 1024 (d) -> N * 512 (k)
        question_features = self.fc_questions(vq)
            
        # N * 512 -> N * 1 * 512 get question_features into same dimensionality as the image
        question_features = question_features.unsqueeze(dim=1)

        # N * 196 * 512 - addition between a matrix and a vector is performed by adding each column of the matrix by the vector, e.g. image matrix (k,m) and question vector (k,1)
        ha = self.tanh(question_features + img_features)
        
        # add dropout
        if hasattr(self, 'dropout'):
            ha = self.dropout(ha)
        
        # N * 196 (m) * 1 -> N * 196 - given the image feature matrix `img_features` and the question vector `question_features` feed through a single layer neural network with softmax function to generate the attention distribution over the regions of the image
        attention_distribution = self.softmax(self.attention_layer(ha).squeeze(dim=2))

        # N * 196 (m) -> N * 196 (m) * 1
        attention_distribution = attention_distribution.unsqueeze(dim=2)

        # N * 196 * 1 x N * 196 * 1024 -> N * 1024 - based on the attention distribution, we calculate the weighted sum of the image vectors from each region
        filtered_image_features = torch.sum((attention_distribution * vi), dim=1)

        #  N * 1024 + N * 1024 - combine filtered image features with question vector to form a refined query vector 
        refined_query_vector = filtered_image_features + vq

        # instead of combining question vector and global image vector, we use attention to construct a more informative image matrix with higher wegihts on visual regions that are more relevant to the question
        return refined_query_vector

class SAN(pl.LightningModule):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering paper (Zhou et al, 2017).
    """
    def __init__(self, question_vocab_size, ans_vocab_size=1000, word_embed_size=300, hidden_size=512, num_layers=2, embed_size=1024, feature_extract=True, num_attention_layer=2, dropout=0.5):
        super(SAN, self).__init__()

        # only use the visual features from the last pooling layer since it retains spatial information of the original images
        self.cnn = models.vgg19(pretrained=True).features 

        # freeze all layers if using CNN as feature extractor
        set_parameter_requires_grad(self.cnn, feature_extract)
        
        # linear layer for image channel
        self.fc_image = nn.Linear(hidden_size,embed_size)      
                
        # lstm
        self.word2vec = nn.Embedding(question_vocab_size, word_embed_size, padding_idx=VQA.questions_vocabulary.word2idx(Vocabulary.PAD_TOKEN))
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers, batch_first=True)
        self.fc_questions = nn.Linear(2*num_layers*hidden_size, embed_size)

        # attention networks
        self.attention_layers = nn.ModuleList([Attention(embed_size, hidden_size)] * num_attention_layer)

        # classifier
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        # self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

        # activation funcitons
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        
        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # initialize parameters for fc layers
        weights_init(self.fc_image)
        weights_init(self.lstm)
        weights_init(self.word2vec)
        weights_init(self.fc_questions)
        weights_init(self.fc1) 
        # weights_init(self.fc2) 

        # save hyperparameters
        self.save_hyperparameters()

    def forward(self, image, question_indices):
        """ 
        Since we are using Pytorch Lightning, the forward method defines how the LightningModule behaves during inference/prediction. 
        """
        #####################
        ### IMAGE CHANNEL ###
        #####################
        # N * 3 * 448 * 448 -> N * 512 * 14 * 14 - 14 × 14 is the number of regions in the image and 512 is the dimension of the feature vector for each region
        img_features = self.cnn(image) # each pixel corresponds to a 32 × 32 pixel region of the original input image and we have 196 feature vectors for each image region

        # N * 512 * 14 * 14 -> N * 196 * 512
        img_features = img_features.reshape(img_features.size(0), -1, img_features.size(1))

        # N * 196 * 512 -> N * 196 * 1024 - use a single layer perceptron to transform each feature vector to a new vector that has the same dimension as the question vector
        img_features = self.leaky_relu(self.fc_image(img_features))
                
        ########################
        ### QUESTION CHANNEL ###
        ########################
        # getting embedding from embedding layer
        question_embeddings = self.tanh(self.word2vec(question_indices))
        
        # question embedding obtained from the LSTM is a concatenation of last cell state and last hidden state representations from each of the hidden layers
        _, (hidden, cell) = self.lstm(question_embeddings)
        
        # concatenate cell state and hidden state of last layer
        question_features = torch.cat((hidden, cell), dim=2)

        # transpose since output of the LSTM has batch size on second dimension
        question_features = torch.transpose(question_features, 0, 1).reshape(question_embeddings.size(0), -1)

        # N * 2048 -> N * 1024
        u = self.leaky_relu(self.fc_questions(question_features))

        ##########################
        ### ATTENTION NETWORKS ###
        ##########################
        for attention_layer in self.attention_layers:
            u = attention_layer(img_features, u)

        ##################
        ### CLASSIFIER ###
        ##################
        # filtered_features = self.dropout(u)       
        logits = self.fc1(u)       
        # filtered_features = self.leaky_relu(filtered_features)
        # filtered_features = self.dropout(filtered_features)
        # logits = self.fc2(filtered_features)

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

        # TODO compute val_acc using q_ids and answers similar to basic_vqa

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
    	shuffle=False,
    	num_workers=num_workers,
        collate_fn=VQA.vqa_collate
	)

    model = SAN(
        question_vocab_size=VQA.questions_vocabulary.size,
        ans_vocab_size=VQA.answers_vocabulary.size
        )

    train(model, train_dataloader, val_dataloader, epochs)
