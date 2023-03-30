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


from datasets.embryo_public import get_public_embryo
from arguments import args_parser
# from trainers import Trainer
import argparse 

parser = args_parser()
args = parser.parse_args()
print(args)

# Default path for Sonya
# local_path = "/media/nyuad/189370B3586E6F7C/group1"
local_path = args.path

em_train_dl, em_val_dl, em_test_dl = get_public_embryo(args)


if args.model == 'resnet18':
    resnet18 = models.resnet18(models.ResNet18_Weights.IMAGENET1K_V1).to('cuda')
   
    optimizer = optim.Adam(resnet18.parameters(), lr=0.1, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.01, patience=5)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 50
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Freeze all the layers in the model
    for param in resnet18.parameters():
        param.requires_grad = False

    # Create a new output layer with 16 output units
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 16).to("cuda")
    resent18 = resnet18.to(device)

# elif args.model == 'transformer':
#     trainer = DAFTTrainer(train_dl, 
#         em_val_dl, 
#         args,
#         test_dl = em_test_dl)
else:
    raise ValueError("not Implementation for args.model")

if args.mode == 'train':
    print("==> training")
    trainer.train()
elif args.mode == 'eval':
    trainer.eval()
else:
    raise ValueError("not Implementation for args.mode")