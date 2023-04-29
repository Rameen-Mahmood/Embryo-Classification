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
import matplotlib.pyplot as plt


train_loss_values = []
val_loss_values = []


class Trainer():
    def __init__(self, args, dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=50):
        self.args = args
        self.time_start = time.time()
        self.time_end = time.time()
        self.epoch = 0
        #self.best_auroc = 0.0

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
        self.epochs_stats = {'loss train': self.train_loss_values, 
            'loss val': self.val_loss_values, 
            'auroc train': self.train_auroc_values,
            'auroc val': self.val_auroc_values}


    def load_state(self):
        if self.args.load_state is None:
            return

        path = f'checkpoints/{self.args.load_state}.pth.tar'
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


    def save_checkpoint(self, prefix='best'):
        path = f'{self.args.save_dir}/{prefix}_checkpoint.pth.tar'
        torch.save(
            {
            'epoch': self.epoch, 
            'state_dict': self.model.state_dict(), 
            #'best_auroc': self.best_auroc, 
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

        val_auroc_scores = []
        for i in range(len(self.val_auroc_values)):
            val_auroc_scores.append(self.val_auroc_values[i].item())
        plt.plot(np.array(val_auroc_scores), 'y', label='Val')
       

        train_auroc_scores = []
        for i in range(len(self.train_auroc_values)):
            train_auroc_scores.append(self.train_auroc_values[i].item())
        plt.plot(np.array(train_auroc_scores), 'b', label='Train')

        plt.title('Training and Validation AUROC')
        plt.xlabel('Epochs')
        plt.ylabel('AUROC')
        plt.legend(loc='best')

        plt.savefig(f"{self.args.save_dir}/{filename}")
        plt.close()


    def plot_loss(self):
        filename = "loss_curves.pdf"
        
        val_loss_values = []
        for i in range(len(self.val_loss_values)):
            val_loss_values.append(self.val_loss_values[i])
        plt.plot(np.array(val_loss_values), 'y', label='Val')


        train_loss_values = []
        for i in range(len(self.train_loss_values)):
            train_loss_values.append(self.train_loss_values[i])
        plt.plot(np.array(train_loss_values), 'b', label='Train')

        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')

        plt.savefig(f"{self.args.save_dir}/{filename}")
        plt.close()


    def test_net(self):
        test_loss = 0.0
        class_correct = list(0 for i in range(16))
        class_total = list(0 for i in range(16))
        self.model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for data, target in tqdm(self.dataloaders['test']):
            data, target = data.to(device), target.to(device)
            with torch.no_grad(): # turn off autograd for faster testing
                output = self.model(data)
                loss = self.criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())

            if len(target) == self.args.batch_size: 
                for i in range(self.args.batch_size):
                    label = target.data[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1

        test_loss = test_loss / self.dataset_sizes['test']
        print('Test Loss: {:.4f}'.format(test_loss))
        classes = ["tPB2","tPNa","tPNf","t2","t3","t4", "t5", "t6", "t7","t8", "t9+","tM","tSB", "tB","tEB","tHB"]
        for i in range(16):
            if class_total[i] > 0:
                print("Test Accuracy of %5s: %2d%% (%2d/%2d)" % (
                    classes[i], 100*class_correct[i]/class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])
                ))
            else:
                print("Test accuracy of %5s: NA" % (classes[i]))
        
        print("Test Accuracy of %2d%% (%2d/%2d)" % (
                    100*np.sum(class_correct)/np.sum(class_total), np.sum(class_correct), np.sum(class_total)
                ))



    def train_epoch(self):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        best_acc = 0.0
        
        metric = MulticlassAUROC(num_classes=16, average="macro", thresholds=None)
        for self.epoch in range(self.num_epochs):
            print(f'Epoch {self.epoch}/{self.num_epochs - 1}')
            print("-"*10)
            
            for phase in ['train', 'val']: # We do training and testing phase per epoch
                if phase == 'train':
                    self.model.train() # model to training mode
                else:
                    self.model.eval() # model to evaluate
                
                running_loss = 0.0
                running_corrects = 0.0
                auroc_sum = 0.0

                for inputs, labels in tqdm(self.dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    self.optimizer.zero_grad() # ensure that gradients from previous iterations aren't accumulated & only gradients of current iteration are computed
                    
                    with torch.set_grad_enabled(phase == 'train'): # no autograd makes testing go faster
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1) # used for accuracy, selects  prediction with the highest value from model's output tensor
                        train_loss = self.criterion(outputs, labels) #loss bw outputs and labels
                        auroc = metric(outputs, labels)
                        # auroc should be summed for the entire epoch or calculate the min/max
                        if phase == 'train':
                            train_loss.backward()
                            self.optimizer.step()
                    running_loss += train_loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    auroc_sum += auroc * inputs.size(0)
                
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_auroc = auroc_sum / self.dataset_sizes[phase]

                if phase == 'train':
                    self.train_loss_values.append(epoch_loss)
                    self.train_auroc_values.append(epoch_auroc)
                    #self.scheduler.step() # step at end of epoch

                else: 
                    self.save_checkpoint(prefix = 'last') #best
                    self.val_loss_values.append(epoch_loss)
                    self.val_auroc_values.append(epoch_auroc)

                
                epoch_acc =  running_corrects.double() / self.dataset_sizes[phase]

                print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
                
                if phase == 'val' and epoch_acc > best_acc:
                    self.save_checkpoint() #best
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict()) # keep the best testing accuracy model
            print()
            
        print('Finished Training Trainset')

        time_elapsed = time.time() - since # slight error
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("Best Val Acc: {:.4f}".format(best_acc))
        
        self.model.load_state_dict(best_model_wts)
        return self.model
