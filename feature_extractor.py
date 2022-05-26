# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from blocks import TransformerEncoder, Tokenizer

class Extractor(nn.Module):
    def __init__(self,
                 embedding_dim=256,
                 n_input_channels=3,
                 token_dim_reduce=4,
                 *args, **kwargs):
        super(Extractor, self).__init__()
        self.embedding_dim = embedding_dim
        self.token_dim_reduce = token_dim_reduce
        
        self.conv1 = self.__build_conv(3, 64) #(64, 112,112) **
        self.conv2 = self.__build_conv(64, 128) #(128, 56, 56) **
        self.conv3 = self.__build_conv(128, 256) #(256, 28, 28) **
        self.conv4 = self.__build_conv(256, 512) #(512, 14, 14) **
        
        self.apply(self.init_weight)
        
        self.tokenizer1 = self.__build_tokenizer(64, 8)
        self.tokenizer2 = self.__build_tokenizer(128, 4)
        self.tokenizer3 = self.__build_tokenizer(256, 2)
        self.tokenizer4 = self.__build_tokenizer(512, 1) 
        #(B, 14*14, C) 
        
        self.transformer = self.__build_transformer(*args, **kwargs)
        
    def forward(self, x):
        map1 = self.conv1(x)
        map2 = self.conv2(map1)
        map3 = self.conv3(map2)
        map4 = self.conv4(map3)
        
        #(B, C)
        seq1 = self.tokenizer1(map1)
        seq2 = self.tokenizer2(map2)
        seq3 = self.tokenizer3(map3)
        seq4 = self.tokenizer4(map4)
        
        seq = torch.cat((seq1, seq2, seq3, seq4), dim=-1) #(B, 14*14, C*4)
        
        return self.transformer(seq) #(B, L, 1+14*14, C) 

        
    def __build_conv(self, in_dim, out_dim):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                             nn.BatchNorm2d(out_dim),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def __build_tokenizer(self, n_channels, patch_size):
        return Tokenizer(n_input_channels = n_channels,
                         n_output_channels = self.embedding_dim//self.token_dim_reduce,
                         kernel_size = patch_size, stride = patch_size, padding = 0)
    
    def __build_transformer(self, *args, **kwargs):
        return TransformerEncoder(embedding_dim = self.embedding_dim,
                                  cls_token = True,
                                  dropout_rate=0.1, # => 0.1
                                  attention_dropout=0.1,
                                  stochastic_depth_rate=0.1,
                                  *args, **kwargs)
    
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)



def Extractor_base():
    return Extractor(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256)
