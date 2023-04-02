import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime, timedelta
import time
import copy
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from torchmetrics.classification import MulticlassAUROC


train_loss_values = []
val_loss_values = []


class Trainer():
    def __init__(self, args, dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=50):
        self.args = args
        self.time_start = time.time()
        self.time_end = time.time()
        self.start_epoch = 1
        self.patience = 0

        self.train_loss_values = []
        self.val_loss_values = []
        self.train_auroc_values = []
        self.val_auroc_values = []

        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs

    def train(self):
        pass

    def validate(self):
        pass

    def load_state(self):
        if self.args.load_state is None:
            return
        checkpoint = torch.load(self.args.load_state)


        own_state = self.model.state_dict()

        for name, param in checkpoint['state_dict'].items():
            if name not in own_state:
                # print(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
        print(f'loaded model checkpoint from {self.args.load_state}')


    def freeze(self, model):
        for p in model.parameters():
           p.requires_grad = False

    def plot_array(self, array, disc='loss'):
        plt.plot(array)
        plt.ylabel(disc)
        plt.savefig(f'{disc}.pdf')
        plt.close()
    
    def computeAUROC(self, y_true, predictions, verbose=1):
        y_true = np.array(y_true)
        predictions = np.array(predictions)

        auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
        ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                            average="micro")
        ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                            average="macro")
        ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                                average="weighted")

        auprc = metrics.average_precision_score(y_true, predictions, average=None)

        auc_scores = []
        auprc_scores = []
        ci_auroc = []
        ci_auprc = []
        if len(y_true.shape) == 1:
            y_true = y_true[:, None]
            predictions = predictions[:, None]
        for i in range(y_true.shape[1]):
            df = pd.DataFrame({'y_truth': y_true[:, i], 'y_pred': predictions[:, i]})
            (test_auprc, upper_auprc, lower_auprc), (test_auroc, upper_auroc, lower_auroc) = get_model_performance(df)
            auc_scores.append(test_auroc)
            auprc_scores.append(test_auprc)
            ci_auroc.append((lower_auroc, upper_auroc))
            ci_auprc.append((lower_auprc, upper_auprc))
        
        auc_scores = np.array(auc_scores)
        auprc_scores = np.array(auprc_scores)
       
        return { "auc_scores": auc_scores,
            
            "auroc_mean": np.mean(auc_scores),
            "auprc_mean": np.mean(auprc_scores),
            "auprc_scores": auprc_scores, 
            'ci_auroc': ci_auroc,
            'ci_auprc': ci_auprc,
            }


    def step_lr(self, epoch):
        step = self.steps[0]
        for index, s in enumerate(self.steps):
            if epoch < s:
                break
            else:
                step = s

        lr = self.args.lr * (0.1 ** (epoch // step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def save_checkpoint(self, prefix='best'):
        path = f'{self.args.save_dir}/{prefix}_checkpoint.pth.tar'
        torch.save(
            {
            'epoch': self.epoch, 
            'state_dict': self.model.state_dict(), 
            'best_auroc': self.best_auroc, 
            'optimizer' : self.optimizer.state_dict(),
            'epochs_stats': self.epochs_stats
            }, path)
        print(f"saving {prefix} checkpoint at epoch {self.epoch}")

    def plot_stats(self, key='loss', filename='training_stats.pdf'):
        for loss in self.epochs_stats:
            if key in loss:
                plt.plot(self.epochs_stats[loss], label = f"{loss}")
        
        plt.xlabel('epochs')
        plt.ylabel(key)
        plt.title(key)
        plt.legend()
        plt.savefig(f"{self.args.save_dir}/{filename}")
        plt.close()


    def plot_auroc(self):
        filename = "auroc_scores.pdf"
        auroc_scores = []
        for i in range(len(self.val_auroc_values)):
            auroc_scores.append(self.val_auroc_values[i].item())
        plt.plot(np.array(auroc_scores), 'r')
        plt.savefig(f"{self.args.save_dir}/{filename}")


    def plot_loss(self):
        filename = "loss_curves.pdf"
        loss_values = []
        for i in range(len(self.val_loss_values)):
            loss_values.append(self.val_loss_values[i])
        plt.plot(np.array(loss_values), 'r')
        plt.savefig(f"{self.args.save_dir}/{filename}")


    def train_epoch(self):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        best_acc = 0.0
        
        metric = MulticlassAUROC(num_classes=16, average="macro", thresholds=None)
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print("-"*10)
            
            for phase in ['train', 'val']: # We do training and testing phase per epoch
                if phase == 'train':
                    self.model.train() # model to training mode
                else:
                    self.model.eval() # model to evaluate
                
                running_loss = 0.0
                running_corrects = 0.0
                
                for inputs, labels in tqdm(self.dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    self.optimizer.zero_grad() # ensure that gradients from previous iterations aren't accumulated & only gradients of current iteration are computed
                    
                    with torch.set_grad_enabled(phase == 'train'): # no autograd makes testing go faster
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1) # used for accuracy, selects  prediction with the highest value from model's output tensor
                        train_loss = self.criterion(outputs, labels) #loss bw outputs and labels
                        auroc = metric(outputs, labels)
                        if phase == 'train':
                            train_loss.backward()
                            self.optimizer.step()
                    running_loss += train_loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / self.dataset_sizes[phase]

                if phase == 'train':
                    self.train_loss_values.append(epoch_loss)
                    self.train_auroc_values.append(auroc)
    #                 scheduler.step() # step at end of epoch
                else: 
                    self.val_loss_values.append(epoch_loss)
                    self.val_auroc_values.append(auroc)

                
                epoch_acc =  running_corrects.double() / self.dataset_sizes[phase]

                print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
                
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict()) # keep the best testing accuracy model
            print()
            
        print('Finished Training Trainset')
                
        plt.plot(np.array(self.train_loss_values), 'r')
        plt.plot(np.array(self.train_auroc_values), 'r')
        time_elapsed = time.time() - since # slight error
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("Best Val Acc: {:.4f}".format(best_acc))
        
        self.model.load_state_dict(best_model_wts)
        return self.model
