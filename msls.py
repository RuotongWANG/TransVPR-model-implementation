# -*- coding: utf-8 -*-

import json
from os.path import join

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms  as transforms
import torch.utils.data as data

default_cities = {
    'train': ["trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
              "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
              "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman", "nairobi", "manila"],
    'toy' :["london"],
    'val': ["cph", "sf"],
    'test': ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"]
}

root_dir = './data/MSLS/'

def transform(img_size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def origin_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def collate_fn(batch):
    batch=list(filter(lambda x: x is not None,batch))
    if len(batch) == 0: return None, None, None, None

    query, positive, negatives = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    
    return query, positive, negatives, negCounts


class MSLS_TestSet(Dataset):
    def __init__(self, img_size=None):
        if img_size:
            self.input_transform = transform(img_size)
        else:
            self.input_transform = origin_transform()
            
        q_file = join(root_dir, 'test_q_sub_all.json')
        db_file = join(root_dir, 'test_db_sub_all.json')
            
        with open(db_file) as f:
            File = json.load(f)
            self.database = File['dbPath']
        with open(q_file) as f:
            File = json.load(f)
            self.query = File['qPath']
            
        self.q_offset = len(self.database)
        self.images = self.database + self.query
        self.images = [join(root_dir, im) for im in self.images]
            
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = self.input_transform(img)
        return img, index

class MSLS_ValSet(Dataset):
    def __init__(self, img_size=None):
        if img_size:
            self.input_transform = transform(img_size)
        else:
            self.input_transform = origin_transform()
            
        q_file = join(root_dir, 'val_q_sub_all.json')
        db_file = join(root_dir, 'val_db_sub_all.json')
            
        with open(db_file) as f:
            File = json.load(f)
            self.database = File['dbPath']
        with open(q_file) as f:
            File = json.load(f)
            self.query = File['qPath']
            
        self.q_offset = len(self.database)
        self.images = self.database + self.query
        self.images = [join(root_dir, im) for im in self.images]
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = self.input_transform(img)
        return img, index