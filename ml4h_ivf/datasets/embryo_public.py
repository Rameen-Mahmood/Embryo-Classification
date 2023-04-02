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
 

class Embryo_Public(Dataset):
    def __init__(self, dataframe, args, transform=None):
        self.dataframe = dataframe
        self.args = args
        self.transform = transform

    def __getitem__(self, index):
        img = self.dataframe.iloc[index]['Images']
        label = self.dataframe.iloc[index]['Phase']
        label_id = label_to_id[label]
        
        if self.transform:
            # img = self.transform(img)
            # Load the image and apply the transform
            img_pil = PIL.Image.open(img).convert("RGB")
            image_tensor = self.transform(img_pil)
            
        return image_tensor, label_id
    
    def __len__(self):
        return len(self.dataframe)


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
		df = pd.DataFrame(data=images, columns=['Images'])
		df['Index'] = frame_num
	    # pd.options.display.max_colwidth = 150
		img_df[sample_str.split('/')[-1]] = df
	return img_df


def merge_df(all_df_img, all_df_ann):
	dataframes = []
	for x in all_df_img:
		if x == '.DS_Store':
		   continue
		else:
			# print(all_df_img.keys())
			mapped_df = pd.merge(all_df_img[x], all_df_ann[x], on='Index')
			dataframes.append(mapped_df)
	meta_dataset = pd.concat(dataframes, axis=0)
	return meta_dataset


def get_transforms(args):
    train_transforms = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
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


def get_public_embryo(args):

	train_transforms, test_transforms = get_transforms(args)
	
	local_path = args.path
	
	annotations_path = local_path + "/embryo_dataset_annotations"
	annotations = [file for file in Path(annotations_path).glob('*.csv')]
	ann_df = make_ann_df(annotations)
	
	imgs_path = local_path + "/embryo_dataset_F15"
	imgs_dir = [img for img in Path(imgs_path).glob('*')]
	img_df = make_img_df(imgs_dir)
	
	dataset = merge_df(img_df, ann_df)
	
	train_df, test_df = train_test_split(dataset, test_size=0.15)
	train_df, val_df = train_test_split(train_df, test_size=0.15)
	
	
	train_dataset = Embryo_Public(train_df, args, transform=train_transforms)
	val_dataset = Embryo_Public(val_df, args, transform=test_transforms)
	test_dataset = Embryo_Public(test_df, args, transform=test_transforms)
	
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
	
	
	return train_loader, val_loader, test_loader
