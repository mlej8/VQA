import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning.logging import CometLogger
from pytorch_lightning.utilities import seed_everything

from pretrained import initialize_model
from datetime import datetime

from utils import weights_init

from simple_baseline_vqa_params import *

# setting the seed for reproducability (it is important to set seed when using DPP mode)
seed_everything(7)

class SimpleBaselineVQA(pl.LightningModule):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering paper (Zhou et al, 2017).
    """
    def __init__(self, num_questions, num_answers):
        super(SimpleBaselineVQA, self).__init__()
        
        # the output size of Imagenet is 1000 and we want to resize it to 1024
        self.googlenet = initialize_model("googlenet", 1024, True, use_pretrained=True) # TODO question: are we feature extracting or finetuning ? we can try both.. 
        self.fc_questions = nn.Linear(num_questions, 1024)
        self.fc2 = nn.Linear(2048, num_answers)
        
        self.criterion = nn.NLLLoss()
        self.log = CometLogger( 
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
        img_features = self.googlenet(image)

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
        (image, question_encodings), labels = batch
        
        # get predictions using forward method 
        preds = self(image, question_encodings)
        
        # logging training loss
        self.log("train_loss", loss)

        # compute NLL loss
        loss = self.criterion(preds,labels)

        return loss
    
    def configure_optimizers(self):
        """ 
        Configure our optimizers.
        """

        # creating optimizer for our model
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        return optimizer