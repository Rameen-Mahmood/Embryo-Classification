import os
import numpy as np
from PIL import ImageFile
import PIL.ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd 
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from pathlib import Path

planes = ["embryo_dataset", "embryo_dataset_F15", "embryo_dataset_F30", "embryo_dataset_F45",
    "embryo_dataset_F-15", "embryo_dataset_F-30", "embryo_dataset_F-45"]
 

class Embryo_Public(Dataset):
    def __init__(self, dataframe, args, transform=None):
        self.dataframe = dataframe
        self.args = args
        self.transform = transform
        # Define the mapping from phase class indices to labels
        self.id_to_label = {0:"tPB2", 1:"tPNa",
                       2:"tPNf", 3:"t2", 
                       4:"t3", 5:"t4", 
                       6:"t5", 7:"t6", 
                       8:"t7", 9:"t8", 
                       10:"t9+", 11:"tM", 
                       12:"tSB", 13:"tB", 
                       14:"tEB", 15:"tHB"}

        self.label_to_id = {"tPB2":0, "tPNa":1,
                       "tPNf":2, "t2":3, 
                       "t3":4, "t4":5, 
                       "t5":6, "t6":7, 
                       "t7":8, "t8":9, 
                       "t9+":10, "tM":11, 
                       "tSB":12, "tB":13, 
                       "tEB":14, "tHB":15}

    def __getitem__(self, index):
        img = self.dataframe.iloc[index]["Image"]
        label = self.dataframe.iloc[index]["Phase"]
        label_id = self.label_to_id[label]
        time_stamp = self.dataframe.iloc[index]["TimeStamp"]
        ttd = self.dataframe.iloc[index]["TTD"]

        if self.transform:
            # img = self.transform(img)
            # Load the image and apply the transform
            img_pil = PIL.Image.open(img).convert("RGB")
            image_tensor = self.transform(img_pil)
            
        return image_tensor, label_id, time_stamp, ttd
    
    def __len__(self):
        return len(self.dataframe)


def make_time_df(timestamps):
    time_df={}
    for stamp_file in timestamps:
        time_str = str(stamp_file)
        colnames = ['Index', 'TimeStamp']
        d = pd.read_csv(time_str, names=colnames, header=0)
        df = pd.DataFrame(data = d)

        # getting index of substrings for sample name, i.e. AA83-7
        idx1 = time_str[::-1].index('/')
        idx2 = time_str[::-1].index('_')
        
        sample_name = time_str[len(time_str)-idx1: len(time_str)-idx2-1]       
        time_df[sample_name] = df
    return time_df


def make_ann_df(annotations):
    ann_df = {}
    for ann in annotations:
        ann_str = str(ann)
        colnames = ['label', 'start_time', 'end_time']
        d = pd.read_csv(ann_str,names=colnames, header=None)
        df = pd.DataFrame(data = d)
        
        phases = []
        for index, row in df.iterrows():
            label = row['label']
            start_time = int(row['start_time'])
            end_time = int(row['end_time'])
            phases.append((label, start_time, end_time))
            
        labels = []
        
        for phase in phases:
            label, start_time, end_time = phase
            label_list = [label] * (end_time - start_time + 1)
            labels.extend(label_list)
        index = list(range(1,len(labels)+1))
        
        labels_df = pd.DataFrame(labels, columns=['Phase'])
        labels_df['Index'] = index

        # getting index of substrings for sample name, i.e. AA83-7
        idx1 = ann_str[::-1].index('/')
        idx2 = ann_str[::-1].index('_')
        
        sample_name = ann_str[len(ann_str)-idx1: len(ann_str)-idx2-1]       
        ann_df[sample_name] = labels_df
    return ann_df


def make_img_df(imgs_dir):
    img_df = {}
    for sample in imgs_dir:
        sample_str = str(sample)
        # iterate over files in that directory

        images = [image for image in Path(sample_str).glob('*.jpeg')]
        frame_num = [int(str(image).split('RUN')[1].split('.')[0]) for image in Path(sample_str).glob('*.jpeg')]  
        
        # paths = [path.parts[-3:] for path in Path(image_dir_path).rglob('*.jpg')]
        df = pd.DataFrame(data=images, columns=['Image'])
        df['Index'] = frame_num
        # pd.options.display.max_colwidth = 150
        img_df[sample_str.split('/')[-1]] = df
    return img_df


def merge_df(all_df_img, all_df_ann, all_df_time, all_df_ttb):
    dataframes = []
    for x in all_df_img:
        if x == '.DS_Store':
            continue
        else:
            mapped_df = pd.merge(pd.merge(pd.merge(all_df_img[x], all_df_ann[x], on="Index"), 
                    all_df_time[x], on="Index"), all_df_ttb[x], on="Index")
            
            dataframes.append(mapped_df)
    meta_dataset = pd.concat(dataframes, axis=0)
    return meta_dataset


def get_transforms(args):
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


def get_public_embryo(args, surv = False):

    train_transforms, test_transforms = get_transforms(args)
    
    local_path = args.path
    
    annotations_path = local_path + "/embryo_dataset_annotations"
    annotations = [file for file in Path(annotations_path).glob('*.csv')]
    ann_df = make_ann_df(annotations)
    
    # 7 focal planes
    imgs_path = local_path + "/embryo_dataset_F15"
    imgs_dir = [img for img in Path(imgs_path).glob('*')]
    img_df = make_img_df(imgs_dir)

    # embryo_dataset_time_elapsed
    times_path = local_path + "/embryo_dataset_time_elapsed"
    times_dir = [time for time in Path(times_path).glob('*')]
    time_df = make_time_df(times_dir)

    # add an extra column of time to blastocyst 
    # the first tB timestamp for each sample

    ttb_df = {} # time to blastocsyst dataframe
    colnames = ['Index', 'TTB']
    for sample_name in ann_df:
        tb_start_index = -1
        tb_start_time = 0.0
        
        for index, row in ann_df[sample_name].iterrows(): #row
            if row['Phase'] == 'tB':
                tb_start_index = row['Index']
                break

        for index, row in time_df[sample_name].iterrows():
            if row['Index'] == tb_start_index:
                tb_start_time = row['TimeStamp']
                break
        # df = pd.DataFrame(data = d)
        ttb_df[sample_name] = pd.DataFrame([(tb_start_time-row['TimeStamp']) 
            for index, row in time_df[sample_name].iterrows()], columns=['TTD'])
        ttb_df[sample_name]['Index'] = time_df[sample_name]['Index']

    # merge
    dataset = merge_df(img_df, ann_df, time_df, ttb_df)
    print(dataset.head())

    if(surv == True):
        return dataset

    train_df, test_val_df = train_test_split(dataset, test_size=0.3)
    test_df, val_df = train_test_split(test_val_df, test_size=0.5)
    
    train_dataset = Embryo_Public(train_df, args, transform=train_transforms)
    val_dataset = Embryo_Public(val_df, args, transform=test_transforms)
    test_dataset = Embryo_Public(test_df, args, transform=test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    

    train_size =len(train_dataset)
    val_size =len(val_dataset)
    test_size = len(test_dataset)

    return train_loader, val_loader, test_loader, train_size, val_size, test_size
