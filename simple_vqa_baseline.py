import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning.logging import CometLogger
from pytorch_lightning.utilities import seed_everything

from pretrained import initialize_model
from datetime import datetime

from utils import weights_init

from params.simple_vqa_baseline import *
from config import *

from preprocessing.vocabulary import Vocabulary

from train import train

# setting the seed for reproducability (it is important to set seed when using DPP mode)
seed_everything(7)

class SimpleBaselineVQA(pl.LightningModule):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering paper (Zhou et al, 2017).
    """
    def __init__(self, questions_vocab_size, answers_vocab_size=1000):
        super(SimpleBaselineVQA, self).__init__()
        
        # the output size of Imagenet is 1000 and we want to resize it to 1024
        self.googlenet, self.input_size = initialize_model("googlenet", 1024, True, use_pretrained=True) # TODO try feature extracting vs finetuning
        self.fc_questions = nn.Linear(questions_vocab_size, 1024)
        self.fc2 = nn.Linear(2048, answers_vocab_size)
        
        # using negative log likelihood as loss
        self.criterion = nn.NLLLoss()
        self.logger = CometLogger( 
                    save_dir="logs/",
                    workspace="vqa",
                    project_name="simple_baseline_vqa_by_fb", 
                    experiment_name="simple_baseline_vqa_{}".format(datetime.now().strftime("%b_%d_%H_%M_%S")) 
                )

        # initialize parameters for fc layers
        weights_init(fc2)
        weights_init(fc_questions)

    def forward(self, image, question_encodings):
        """ 
        Since we are using Pytorch Lightning, the forward method defines how the LightningModule behaves during inference/prediction. 
        """
        
        # one hot encode questions
        question_encodings, _ = torch.max(question_encodings, 1)
        
        # getting visual features
        img_feat = self.googlenet(image)

        # getting language features
        ques_features = self.fc_questions(question_encodings)

        # concatenate features        
        features = torch.cat((img_feat, ques_features), 1)

        # one fully connected layer
        output = self.fc2(features)

        return F.softmax(output)
    
    def training_step(self, batch):
        """ 
        training_step method defines a single iteration in the training loop. 
        """

        # The LightningModule knows what device it is on - you can reference via `self.device`, it makes your models hardware agnostic (you can train on any number of GPUs spread out on differnet machines)
        (image, question_encodings, answers) = batch
        
        # get predictions using forward method 
        preds = self(image, question_encodings)
        
        # compute NLL loss
        loss = self.criterion(preds, answers)
        
        # logging training loss
        self.logger("train_loss", loss)

        return loss

    def training_step(self, batch):
        """ 
        validation_step method defines a single iteration in the validation loop. 
        """

        # The LightningModule knows what device it is on - you can reference via `self.device`
        (image, question_encodings, answers) = batch
        
        # get predictions using forward method 
        preds = self(image, question_encodings)
        
        # compute NLL loss
        loss = self.criterion(preds, answers)
        
        # logging validation loss
        self.logger("validation_loss", loss)

        return loss
    
    def configure_optimizers(self):
        """ 
        Configure our optimizers.
        """
        # creating optimizer for our model
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        return optimizer

if __name__ == "__main__":
    train(SimpleBaselineVQA(num_questions=1024, num_answers=10), epochs)