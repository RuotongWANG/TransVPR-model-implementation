# -*- coding: utf-8 -*
import numpy as np
from PIL import Image
import torch
import torchvision.transforms  as transforms

from feature_extractor import Extractor_base
from blocks import POOL

def transform(img_size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

if __name__ == "__main__":
    ckpt = "CHECKPOINT"
    img_file = "IMAGE"
    
    img_size = np.array([480,640])
    N_patch = img_size//(2**4)
    input_transform = transform(img_size)
    img = Image.open(img_file)
    img = input_transform(img)
    
    model = Extractor_base()
    pool = POOL(model.embedding_dim)
    model.add_module('pool', pool)
    
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint)
    
    patch_feat = model(input)
    global_feat, attention_mask = model.pool(patch_feat)  
    
