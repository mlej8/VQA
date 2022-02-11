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

from params.strong_baseline import *
from params.scheduler import *

from vqa import VQA
from preprocessing.vocabulary import Vocabulary

from pretrained import set_parameter_requires_grad

class Attention(torch.nn.Module):
    def __init__(self, v_features, q_features, mid_features=1024, glimpses=2, drop=0.5):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x

    def apply_attention(self, input, attention):
        """ Apply any number of attention maps over the input. """
        n, c = input.size()[:2]
        glimpses = attention.size(1)

        # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
        input = input.view(n, 1, c, -1) # [n, 1, c, s]
        attention = attention.view(n, glimpses, -1)
        attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
        weighted = attention * input # [n, g, v, s]
        weighted_mean = weighted.sum(dim=-1) # [n, g, v]
        return weighted_mean.view(n, -1)

    def tile_2d_over_nd(self, feature_vector, feature_map):
        """ Repeat the same feature vector over all spatial positions of a given feature map.
            The feature vector should have the same batch size and number of features as the feature map.
        """
        n, c = feature_vector.size()
        spatial_size = feature_map.dim() - 2
        tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
        return tiled


class StrongVQABaseline(pl.LightningModule):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering paper (Zhou et al, 2017).
    """
    def __init__(self, question_vocab_size, ans_vocab_size=1000, word_embed_size=300, hidden_size=512, num_layers=2, embed_size=1024, feature_extract=True, input_size=224):
        super(StrongVQABaseline, self).__init__()
        
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

        # attention
        self.attention = Attention(embed_size, embed_size)
        
        # mlp
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

        # activation funcitons
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        
        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # initialize parameters for fc layers
        weights_init(self.lstm)
        weights_init(self.fc1) 
        weights_init(self.fc2) 
        weights_init(self.fc_questions)
        weights_init(self.fc_image)
        self.attention.apply(weights_init)

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

        # feed img and question channels to attention
        a = self.attention(img_features, question_features)

        # apply attention to visual features
        img_features = self.attention.apply_attention(img_features, a)
    
        # point-wise multiplication
        combined_feature = self.dropout(self.leaky_relu(torch.mul(img_features, question_features)))

        # a fully connected neural network classifier with 2 hidden layers and 1000 hidden units (dropout 0.5) in each layer with tanh non-linearity
        combined_feature = self.fc1(combined_feature)       
        combined_feature = self.leaky_relu(combined_feature)
        combined_feature = self.dropout(combined_feature)
        logits = self.fc2(combined_feature)

        # softmax layer to obtain a distribution over K answers
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
        _, labels = torch.max(answer, dim=1)
        
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
        _, labels = torch.max(answer, dim=1)

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=verbose, factor=factor, patience=patience)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitored_loss}

    def attention(self):
        pass

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
    	num_workers=num_workers
        # , collate_fn=VQA.vqa_collate
	)
    
    val_dataloader = DataLoader(
    	val_dataset,
    	batch_size=batch_size, 
    	shuffle=False,  # set False for validation dataloader
    	num_workers=num_workers
        # , collate_fn=VQA.vqa_collate
	)

    model = StrongVQABaseline(
        question_vocab_size=VQA.questions_vocabulary.size,
        ans_vocab_size=VQA.answers_vocabulary.size
        )

    train(model, train_dataloader, val_dataloader, epochs)
