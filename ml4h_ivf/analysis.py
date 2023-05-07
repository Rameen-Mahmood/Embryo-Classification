import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# For preprocessing
#! pip install sklearn-pandas
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
from sklearn.model_selection import train_test_split

import torch # For building the networks 
from torch import nn
import torch.nn.functional as F
import torchtuples as tt # Some useful functions
import torchvision.transforms as transforms

from pycox.datasets import metabric
from pycox.models import LogisticHazard
from pycox.models.loss import NLLLogistiHazardLoss
from pycox.evaluation import EvalSurv

from PIL import ImageFile
import PIL.ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


from dataset.embryo_public import get_public_embryo
import argparse
from arguments import args_parser



parser = args_parser()
args = parser.parse_args()
print(args)

# df_train = metabric.read_df()
df_train, df_val, df_test = get_public_embryo(args, surv = True)


# feature transform, Image
def get_transforms(df, train = True):
    if train == True:
        transform = transforms.Compose([
            transforms.Resize(224),
            #transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    #df_transform = [print(x) for x in df['Image']]
    df_transform = [transform(PIL.Image.open(x).convert("RGB")) for x in df['Image']]

    return df_transform

x_train = get_transforms(df_train)
x_val = get_transforms(df_val, train = False)
x_test = get_transforms(df_test, train = False)
print("Image transform completed.")

# label transform
num_durations = 10
labtrans = LogisticHazard.label_transform(num_durations)
get_target = lambda df: (df['TimeStamp'].values, 
    (df['Phase'].values == 'tB') or (df['Phase'].values == 'tEB') or (df['Phase'].values == 'tHB'))
y_train_surv = labtrans.fit_transform(*get_target(df_train))
y_val_surv = labtrans.transform(*get_target(df_val))

train = tt.tuplefy(x_train, (y_train_surv, x_train))
val = tt.tuplefy(x_val, (y_val_surv, x_val))

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)

# combined idx_durations and events intro the tuple y_train_surv
print(y_train_surv)

# modify the original transformer to consider "duration"


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


model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts, loss=loss)

dl = model.make_dataloader(train, batch_size=5, shuffle=False)
batch = next(iter(dl))


model.compute_metrics(batch)
model.score_in_batches(*train)

metrics = dict(
    loss_surv = LossAELogHaz(1),
    loss_ae   = LossAELogHaz(0)
)
callbacks = [tt.cb.EarlyStopping()]

batch_size = 256
epochs = 10
log = model.fit(*train, batch_size, epochs, callbacks, False, val_data=val, metrics=metrics)

res = model.log.to_pandas()


#_ = res[['train_loss', 'val_loss']].plot()

#_ = res[['train_loss_surv', 'val_loss_surv']].plot()

#_ = res[['train_loss_ae', 'val_loss_ae']].plot()

# prediction
surv = model.interpolate(10).predict_surv_df(x_test)\

surv.iloc[:, :5].plot(drawstyle='steps-post')
# plt.ylabel('S(t | x)')
# _ = plt.xlabel('Time')
