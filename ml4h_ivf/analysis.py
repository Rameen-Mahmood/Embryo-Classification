import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# For preprocessing
#! pip install sklearn-pandas
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
from sklearn.model_selection import train_test_split

import torch # For building the networks 
from torch import nn
import torch.nn.functional as F
import torchtuples as tt # Some useful functions
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from pycox.datasets import metabric
from pycox.models import LogisticHazard
from pycox.models.loss import NLLLogistiHazardLoss
from pycox.evaluation import EvalSurv

from PIL import ImageFile
import PIL.ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


from dataset.embryo_public import get_public_embryo
from dataset.embryo_public import Embryo_Public
import argparse
from arguments import args_parser

parser = args_parser()
args = parser.parse_args()
print(args)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# df_train = metabric.read_df()
df_train, df_val, df_test = get_public_embryo(args, surv = True)
df_train = df_train.sample(frac=0.002)
df_val = df_val.sample(frac=0.002)
df_test = df_test.sample(frac=0.002)
print("Downsampling completed.", df_train.size)

# feature transform, Image

# df_transform = [transform(PIL.Image.open(x).convert("RGB")) for x in df['Image']]

def get_transform(args):
    train_transforms = transforms.Compose([
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, test_transforms

train_transforms, test_transforms = get_transform(args)


x_train = Embryo_Public(df_train, args, transform=train_transforms)
print("Image transform for training completed.")
x_val = Embryo_Public(df_val, args, transform=train_transforms)
x_test = Embryo_Public(df_test, args, transform=train_transforms)


# label transform
num_durations = 10
labtrans = LogisticHazard.label_transform(num_durations)

get_target = lambda df: (df['TimeStamp'].values, 
    np.array([(x=='tB') or (x=='tEB') or (x=='tHB') for x in df['Phase']]))

y_train_surv = labtrans.fit_transform(*get_target(df_train))
y_val_surv = labtrans.transform(*get_target(df_val))
y_test_surv = labtrans.transform(*get_target(df_test))

# train = tt.tuplefy(x_train, (y_train_surv, x_train))
# val = tt.tuplefy(x_val, (y_val_surv, x_val))

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)

# combined idx_durations and events intro the tuple y_train_surv
print("y_train_surv", type(y_train_surv))
print("out_features: ", labtrans.out_features)

# modify the original transformer to consider "duration"
#in_features = x_train.shape[1]
out_features = labtrans.out_features

transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True, force_reload=True)
for param in transformer.parameters(): #freeze model
    param.requires_grad = False

n_inputs = transformer.head.in_features
transformer.head = nn.Sequential(
    nn.Linear(n_inputs, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, out_features)
)
net = transformer.to(device)

# the loss
class LossAELogHaz(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        assert (alpha >= 0) and (alpha <= 1), 'Need `alpha` in [0, 1].'
        self.alpha = alpha
        self.loss_surv = NLLLogistiHazardLoss()
        self.loss_ae = nn.MSELoss()
        
    def forward(self, phi, decoded, target_loghaz, target_ae):
        idx_durations, events = target_loghaz
        loss_surv = self.loss_surv(phi, idx_durations, events)
        loss_ae = self.loss_ae(decoded, target_ae)
        return self.alpha * loss_surv + (1 - self.alpha) * loss_ae

loss = LossAELogHaz(0.6)
model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)


class EmbryoDatasetTime(Dataset):
    """Simulatied data from MNIST. Read a single entry at a time.
    """
    def __init__(self, dataset, time, event):
        self.dataset = dataset
        self.time, self.event = tt.tuplefy(time, event).to_tensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # if type(index) is not int:
        #     raise ValueError(f"Need `index` to be `int`. Got {type(index)}.")
        img = [self.dataset[i][0] for i in index]
        img = torch.stack(img)
        return img, (self.time[index], self.event[index])

dataset_train = EmbryoDatasetTime(x_train, *y_train_surv)
dataset_val = EmbryoDatasetTime(x_val, *y_val_surv)
dataset_test = EmbryoDatasetTime(x_test, *y_test_surv)

# go over one sample
def collate_fn(batch):
    """Stacks the entries of a nested tuple"""
    return tt.tuplefy(batch).stack()

#dl = DataLoader(train, batch_size=5, shuffle=False, collate_fn = collate_fn)
dl_train = tt.data.DataLoaderBatch(dataset_train, batch_size=32, shuffle=True)
dl_val = tt.data.DataLoaderBatch(dataset_val, batch_size=32, shuffle=False)
dl_test = tt.data.DataLoaderBatch(dataset_test, batch_size=32, shuffle=False)
batch = next(iter(dl_train))

#print(model.compute_metrics(batch))
#print(model.score_in_batches(*dataset_train))

# training
metrics = dict(
    loss_surv = LossAELogHaz(1),
    loss_ae   = LossAELogHaz(0)
)
callbacks = [tt.cb.EarlyStopping(patience=10)]

epochs = 50
verbose = True
batch_size = 32

pred = model.predict(batch[0])
print(pred.shape)
log = model.fit_dataloader(dl_train, epochs, callbacks, verbose, val_dataloader=dl_val)

res = model.log.to_pandas()
#print(res.head())


# Prediction
class EmbryoTimeInput(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = self.dataset[index][0]
        return img

dataset_test_x = EmbryoTimeInput(x_test)
dl_test_x = tt.data.dataloader_input_only(dl_test)
print(next(iter(dl_test_x)).shape)


surv = model.predict_surv_df(dl_test_x)

surv.iloc[:, :10].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time (hr)')
plt.savefig(f"{args.save_dir}/survival.pdf")
plt.close()

# Concordance
ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
print(ev.concordance_td('antolini'))

# Brier Score
time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
ev.brier_score(time_grid).plot()
plt.ylabel('Brier score')
_ = plt.xlabel('Time')
plt.savefig(f"{args.save_dir}/brier_score.pdf")
plt.close()

# Negative binomial log-likelihood
ev.nbll(time_grid).plot()
plt.ylabel('NBLL')
_ = plt.xlabel('Time')
plt.savefig(f"{args.save_dir}/Negative binomial log-likelihood.pdf")
plt.close()
