import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset

import numpy as np
import os
from PIL import ImageFile
import PIL.Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import csv
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import pickle


from dataset.embryo_public import get_public_embryo
from arguments import args_parser
from trainers import Trainer
import argparse 
import timm
from timm.loss import LabelSmoothingCrossEntropy # This is better than normal nn.CrossEntropyLoss

parser = args_parser()
args = parser.parse_args()
print(args)

# Default path for Sonya
# local_path = "/media/nyuad/189370B3586E6F7C/group1"
local_path = args.path

em_train_dl, em_val_dl, em_test_dl, train_size, val_size, test_size = get_public_embryo(args)

dataloaders = {
    "train": em_train_dl,
    "val": em_val_dl,
    "test": em_test_dl
}


dataset_sizes = {
    "train": train_size,
    "val": val_size,
    "test": test_size,
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if args.model == 'resnet18':
    resnet18 = models.resnet18(models.ResNet18_Weights.IMAGENET1K_V1).to('cuda')  
    # Freeze all the layers in the model
    for param in resnet18.parameters():
        param.requires_grad = False

    # Create a new output layer with 16 output units
    # resnet18.fc = nn.Linear(num_ftrs, 16).to("cuda")
    
    n_inputs = resnet18.fc.in_features
    resnet18.fc = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(512, 16),
    )
    resent18 = resnet18.to(device)

    optimizer = optim.Adam(resnet18.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.01, patience=5) #not use
    criterion = LabelSmoothingCrossEntropy()
    criterion = criterion.to(device)
    num_epochs = 50

    trainer = Trainer(args, dataloaders, dataset_sizes, resnet18, criterion, optimizer, scheduler, num_epochs)

elif args.model == 'transformer':
    transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    for param in transformer.parameters(): #freeze model
        param.requires_grad = False

    n_inputs = transformer.head.in_features
    transformer.head = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 16)
    )
    transformer = transformer.to(device)

    num_epochs = 100

    criterion = LabelSmoothingCrossEntropy()
    criterion = criterion.to(device)
    optimizer = optim.Adam(transformer.head.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)

    trainer = Trainer(args, dataloaders, dataset_sizes, transformer, criterion, optimizer, scheduler, num_epochs)

else:
    raise ValueError("not Implementation for args.model")

if args.mode == 'train':
    print("==> training")
    trainer.train_epoch()
    trainer.plot_auroc()
    trainer.plot_loss()

# elif args.mode == 'eval':
#     trainer.eval()
else:
    raise ValueError("not Implementation for args.mode")
