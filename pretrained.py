""" 
Pretrained models (https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)

We are using these pretrained models are feature extractor. 
we start with a pretrained model and only update the final layer weights from which we derive predictions.
We are using the pretrained CNN as fixed feature-extractor. 

- Initialize the pretrained model
- Reshape the final layer(s) to have the same number of outputs as the number of classes in the new dataset
- Define for the optimization algorithm and which parameters we want to update during training
- Run the training step
"""

from __future__ import print_function
from __future__ import division
from train import device
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# log versions
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    """ 
    Handles the training and validation of a given pre-trained model.
    
    :param: model: Pretrained PyTorch model name
    :param: dataloaders: Dictionary of dataloaders with keys "train" and "val"
    :param: criterion: Loss function
    :param: optimizer: optimization algorithm
    :param: num_epochs: number of epochs to train and validate for. 
    :param: is_inception: Boolean flag for when the model is an Inception model. 
                          The is_inception flag is used to accomodate the Inception v3 model, as that architecture uses an auxiliary output and the overall model loss respects both the auxiliary output and the final output

    Returns the best performing model's weights and validation accuracy history.
    """
    
    # tracking training time
    since = time.time()

    # validation acc history
    val_acc_history = []

    # storing best models weights and accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                # send to correct device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'): # only compute gradients in train mode
                    if is_inception and phase == 'train':
                        """ 
                        Special case for inception because in training it has an auxiliary output.
                        In train mode, we calculate the loss by summing the final output and the auxiliary output but in testing we only consider the final output.
                        From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        """
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        # get model outputs and 
                        outputs = model(inputs)

                        # compute loss
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model if val accuracy is better
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # return best model
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    """ 
    Set the requires_grad attribute for parameters of all layers depending on the task, e.g. finetuning vs. feature extraction.
    
    This helper function sets the .requires_grad attribute of the parameters in the model to False when we are feature extracting.
    By default, when we load a pretrained model all of the parameters have .requires_grad=True, which is fine if we are training from scratch or finetuning. 
    However, if we are feature extracting and only want to compute gradients for the newly initialized layer then we want all of the other parameters to not require gradients.
    """
    if feature_extracting: # when we are doing feature_extraction, we "freeze" all layers and only allow the ouput layer's parameters to be updated
        for param in model.parameters():
            param.requires_grad = False
        
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """ 
    Method to initialize pretrained models.
    We are reshaping each network, because the final layer of a CNN model, which is often times an FC layer, has the same number of nodes as the number of output classes in the dataset. 
    Since all of the models have been pretrained on Imagenet, they all have output layers of size 1000, one node for each class.
    The goal here is to reshape the last layer to have the same number of inputs as before and the same number of outputs as the number of classes in the dataset.
    
    When feature extracting, we only want to update the parameters of the last layer, e.g. the layer(s) we are reshaping. 
    Therefore, we do not need to compute the gradients of the parameters that we are not changing, so for efficiency we set the .requires_grad attribute of the parameters for all layers except the last layer(s) to False. 
    This is important because by default, this attribute is set to True. 
    Then, when we initialize the new layer and by default the new parameters have .requires_grad=True so only the new layer's parameters will be updated. 
    When we are finetuning we can leave all of the .required_grad's set to the default of True.
    """
    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18": 
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnet50": 
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnet101": 
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnet152": 
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg11_bn": # VGG11_bn 
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    
    elif model_name == "vgg16": # https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    
    elif model_name == "vgg16_bn":
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    
    elif model_name == "vgg19":
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier[-1] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    
    elif model_name == "vgg19_bn":
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "googlenet":
        model_ft = models.googlenet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception": # NOTE: Inceptionv3 expects (299,299) sized images and has auxiliary output
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit(7)

    return model_ft, input_size

if __name__ == "__main__":
    #######################
    ### Hyperparameters ###
    #######################

    # Number of classes in the dataset
    num_classes = 2

    # Batch size for training (change depending on how much memory you have)
    batch_size = 8

    # Number of epochs to train for
    num_epochs = 15

    # Flag for feature extracting vs finetuning: when True we only update the reshaped layer params, when False we finetune the entire model
    feature_extract = False

    # num of workers
    num_workers = 2

    # learning rate
    lr = 1e-5

    # momentum
    momentum = 0.9


    ################
    ### Datasets ###
    ################

    # transformation for data augmentation
    transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Load the CIFAR10 training and test datasets using torchvision
    cifar_trainset = torchvision.datasets.CIFAR10(root='./cifar10data', train=True, download=True, transform=transforms)
    cifar_testset = torchvision.datasets.CIFAR10(root='./cifar10data', train=False,download=True, transform=transforms)

    # Load the MNIST training and test datasets using torchvision
    mnist_trainset = torchvision.datasets.MNIST(root='./mnistdata', train=True, download=True, transform=transforms)
    mnist_testset = torchvision.datasets.MNIST(root='./mnistdata', train=False, download=True, transform=transforms)

    ###################
    ### Dataloaders ###
    ###################

    # create train/test dataloader 
    trainloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(cifar_testset, batch_size=batch_size,shuffle=True, num_workers=num_workers)

    # define a dictionary of dataloaders 
    dataloaders_dict = {"train": trainloader, "val": testloader}

    ##############
    ### Models ###
    ##############

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet50"
    print(f"Chosen model: {model_name}")

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, len(cifar_trainset.classes), feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    #################
    ### Optimizer ###
    #################

    # Gather the parameters to be optimized/updated in this run.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract: # if we are doing feature extract method, we will only update the parameters that we have just initialized, i.e. the parameters with requires_grad is True.
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else: # if we are finetuning we will be updating all parameters.
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=momentum)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))