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
#from tqdm import tqdm
import numpy as np
from sklearn import metrics
from .utils import get_model_performance


train_loss_values = []
val_loss_values = []
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print("-"*10)

        for phase in ['train', 'val']: # We do training and validation phase per epoch
            if phase == 'train':
                model.train() # model to training mode
            else:
                model.eval() # model to evaluate
            
            running_loss = 0.0
            running_corrects = 0.0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'): # no autograd makes validation go faster
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # used for accuracy
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)         
                running_corrects += torch.sum(preds == labels.data)
                
            #if phase == 'train':
                #scheduler.step() # step at end of epoch
            
            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
                train_loss_values.append(epoch_loss)
                #auroc.append()
            else:
                val_loss_values.append(epoch_loss)
            epoch_acc =  running_corrects.double() / dataset_sizes[phase]
            
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) # keep the best validation accuracy model
        print()
        
    print('Finished Training Trainset')           
    plt.plot(np.array(train_loss_values), 'r')
    time_elapsed = time.time() - since # slight error
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best Val Acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model


class Trainer():
    def __init__(self, args):
        self.args = args
        self.time_start = time.time()
        self.time_end = time.time()
        self.start_epoch = 1
        self.patience = 0

    def train(self):
        pass

    def train_epoch(self):
        pass

    def validate(self):
        pass

    # def load_ehr_pheno(self, load_state):
        
    #     checkpoint = torch.load(load_state)
    #     own_state = self.model.state_dict()

    #     for name, param in checkpoint['state_dict'].items():
    #         if name not in own_state or 'ehr_model' not in name:
    #             # print(name)
    #             continue
    #         if isinstance(param, torch.nn.Parameter):
    #             param = param.data
    #         own_state[name].copy_(param)

    #     print(f'loaded ehr checkpoint from {load_state}')

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

    # def load_cxr_pheno(self, load_state):
    #     checkpoint = torch.load(load_state)

    #     own_state = self.model.state_dict()

    #     for name, param in checkpoint['state_dict'].items():
    #         if name not in own_state or 'cxr_model' not in name:
    #             # print(name)
    #             continue
    #         if isinstance(param, torch.nn.Parameter):
    #             param = param.data
    #         own_state[name].copy_(param)

    #     print(f'loaded cxr checkpoint from {load_state}')


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

    # def get_eta(self, epoch, iter):
    #     # import pdb; pdb.set_trace()
    #     done_epoch = epoch - self.start_epoch
    #     remaining_epochs = self.args.epochs - epoch

    #     iter +=1
    #     self.time_end = time.time()
        
    #     delta = self.time_end - self.time_start
        
    #     done_iters = len(self.train_dl) * done_epoch + iter
        
    #     remaining_iters = len(self.train_dl) * remaining_epochs - iter

    #     delta = (delta/done_iters)*remaining_iters
        
    #     sec = timedelta(seconds=int(delta))
    #     d = (datetime(1,1,1) + sec)
    #     eta = f"{d.day-1} Days {d.hour}:{d.minute}:{d.second}"

    #     return eta
    
    # def get_gt(self, y_ehr, y_cxr):
    #     if 'radiology' in self.args.labels_set :
    #         return y_cxr
    #     else:
    #         return torch.from_numpy(y_ehr).float()

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

        
    # def print_and_write(self, ret, prefix='val', isbest=False, filename='results.txt'):
    #     with open(f"{self.args.save_dir}/{filename}", 'a') as results_file:
    #         if isbest:
                
    #             ci_auroc_all = []
    #             ci_auprc_all = []

    #             if len(ret['auc_scores'].shape) > 0:
                    
    #                 for index, class_auc in enumerate(ret['auc_scores']):
    #                     # line = f'{self.val_dl.dataset.CLASSES[index]: <90} & {class_auc:0.3f} & {ret["auprc_scores"][index]:0.3f} ' 
    #                     line = f'{self.val_dl.dataset.CLASSES[index]: <90} & {class_auc:0.3f}({ret["ci_auroc"][index][1]:0.3f}, {ret["ci_auroc"][index][0]:0.3f}) & {ret["auprc_scores"][index]:0.3f} ({ret["ci_auprc"][index][1]:0.3f}, {ret["ci_auprc"][index][0]:0.3f}) ' 
    #                     ci_auroc_all.append([ret["ci_auroc"][index][0] , ret["ci_auroc"][index][1]])
    #                     ci_auprc_all.append([ret["ci_auprc"][index][0] , ret["ci_auprc"][index][1]])
    #                     print(line)
    #                     results_file.write(line)
                                 
    #                 # for index, class_auc in enumerate(ret['auc_scores']):
    #                 #     ci_auroc_all.append([ret["ci_auroc"][index][0] , ret["ci_auroc"][index][1]])
    #                 #     ci_auprc_all.append([ret["ci_auprc"][index][0] , ret["ci_auprc"][index][1]])
    #                 #     line = f'{self.val_dl.dataset.CLASSES[index]: <90} & CI AUROC ({ret["ci_auroc"][index][1]:0.3f}, {ret["ci_auroc"][index][0]:0.3f})    CI AUPRC ({ret["ci_auprc"][index][1]:0.3f}, {ret["ci_auprc"][index][0]:0.3f}) ' 
    #                 #     print(line)
    #                 #     results_file.write(line)
    #             else:
    #                 ci_auroc_all.append([ret["ci_auroc"][0][0] , ret["ci_auroc"][0][1]])
    #                 ci_auprc_all.append([ret["ci_auprc"][0][0] , ret["ci_auprc"][0][1]])

    #             ci_auroc_all = np.array(ci_auroc_all)
    #             ci_auprc_all = np.array(ci_auprc_all)

    #             auc_scores = ret['auc_scores']
    #             auprc_scores = ret['auprc_scores']

    #             accute_aurocs = np.mean(auc_scores) if self.args.labels_set != 'pheno' else np.mean(auc_scores[self.levels == 'acute'])
    #             mixed_aurocs = np.mean(auc_scores) if self.args.labels_set != 'pheno' else np.mean(auc_scores[self.levels == 'mixed'])
    #             chronic_aurocs = np.mean(auc_scores) if self.args.labels_set != 'pheno' else np.mean(auc_scores[self.levels == 'chronic'])
                
    #             accute_auprc = np.mean(auprc_scores) if self.args.labels_set != 'pheno' else np.mean(auprc_scores[self.levels == 'acute'])
    #             mixed_auprc = np.mean(auprc_scores) if self.args.labels_set != 'pheno' else np.mean(auprc_scores[self.levels == 'mixed'])
    #             chronic_auprc = np.mean(auprc_scores) if self.args.labels_set != 'pheno' else np.mean(auprc_scores[self.levels == 'chronic'])


    #             accute_aurocs_ci = np.mean(ci_auroc_all, axis=0) if self.args.labels_set != 'pheno' else np.mean(ci_auroc_all[self.levels == 'acute'], axis=0)
    #             mixed_aurocs_ci = np.mean(ci_auroc_all, axis=0) if self.args.labels_set != 'pheno' else np.mean(ci_auroc_all[self.levels == 'mixed'], axis=0)
    #             chronic_aurocs_ci = np.mean(ci_auroc_all, axis=0) if self.args.labels_set != 'pheno' else np.mean(ci_auroc_all[self.levels == 'chronic'], axis=0)
                
    #             accute_auprc_ci = np.mean(ci_auprc_all, axis=0) if self.args.labels_set != 'pheno' else np.mean(ci_auprc_all[self.levels == 'acute'], axis=0)
    #             mixed_auprc_ci = np.mean(ci_auprc_all, axis=0) if self.args.labels_set != 'pheno' else np.mean(ci_auprc_all[self.levels == 'mixed'], axis=0)
    #             chronic_auprc_ci = np.mean(ci_auprc_all, axis=0) if self.args.labels_set != 'pheno' else np.mean(ci_auprc_all[self.levels == 'chronic'], axis=0)

    #             # import pdb; pdb.set_trace()

    #             line = f"\n\n\n{prefix}  {self.epoch:<3} best mean auc :{ret['auroc_mean']:0.3f} mean auprc {ret['auprc_mean']:0.3f} \n\n\n\
    #                 CI AUROC ({np.mean(ci_auroc_all[:, 0]):0.3f}, {np.mean(ci_auroc_all[:, 1]):0.3f}) CI AUPRC ({np.mean(ci_auprc_all[:, 0]):0.3f}, {np.mean(ci_auprc_all[:, 1]):0.3f}) \n\n\n \
    #                 AUROC accute {accute_aurocs:0.3f} mixed {mixed_aurocs:0.3f} chronic {chronic_aurocs:0.3f}\n\n\n \
    #                 AUROC accute CI ({accute_aurocs_ci[0]:0.3f}, {accute_aurocs_ci[1]:0.3f}) mixed ({mixed_aurocs_ci[0]:0.3f} , {mixed_aurocs_ci[1]:0.3f}) chronic ({chronic_aurocs_ci[0]:0.3f}, {chronic_aurocs_ci[1]:0.3f})\n\n\n \
    #                 AUPRC accute  {accute_auprc:0.3f} mixed {mixed_auprc:0.3f} chronic {chronic_auprc:0.3f} \n\n\n \
    #                 AUPRC accute CI  ({accute_auprc_ci[0]:0.3f}, {accute_auprc_ci[1]:0.3f}) mixed ({mixed_auprc_ci[0]:0.3f},  {mixed_auprc_ci[1]:0.3f}) chronic ({chronic_auprc_ci[0]:0.3f}, {chronic_auprc_ci[1]:0.3f}) \n\n\n\
    #                 " 
    #             print(line)
    #             results_file.write(line)
    #         else:
    #             line = f"\n\n\n{prefix}  {self.epoch:<3} mean auc :{ret['auroc_mean']:0.6f} mean auprc {ret['auprc_mean']:0.6f}\n\n\n " 
    #             print(line)
    #             results_file.write(line)
