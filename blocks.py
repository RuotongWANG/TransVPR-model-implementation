# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        ##(B, L, 1+14*14, C)  => (B, C)
        x = x[:, -1, 0, :]
        return self.fc(x)

class POOL(nn.Module): 
    def __init__(self, embedding_dim, level=[1,3,5]):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.level = level
        self.fc = nn.Sequential(L2Norm(dim=-1),
                                 nn.Linear(embedding_dim*len(level), 
                                           embedding_dim, bias=True), 
                                 L2Norm(dim=-1))
        self.attention_pool = nn.Linear(embedding_dim*len(level), len(level))
        
    def forward(self, x): #(B, L_all, 1+14*14, C)
        
        x = x[:, self.level, :, :] #(B, L, 1+14*14, C)
        x = x.permute(0,2,1,3).reshape(x.size(0), x.size(2), -1) #(B,1+14*14,C*L)
        x, mask = self.pool1d(x) #(B,C*L)
        x = self.fc(x)
        return x, mask
    
    def pool1d(self, x): #(B, 1+14*14, C) => (B,C)

        x = x[..., 1:, :] #(B,14*14,C) or (B,14*14,C*L)
       
        mask = F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2) # (B, 1, 14*14)/(B, L, 14*14)
        features = []
        for i in range(len(self.level)):
            feature = torch.matmul(mask[:,i:i+1,:], 
                    x[..., i*self.embedding_dim:(i+1)*self.embedding_dim])\
                    .squeeze(-2)
            #(B,1,14*14) mul (B,14*14,C) = (B,1,C) => (B,C)
            features.append(feature)
        x = torch.cat(features, -1) # (B,C*L)
        return x, mask   


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=0,
                 n_input_channels=3,
                 n_output_channels=64):
        super(Tokenizer, self).__init__()

        self.conv_layer = nn.Conv2d(n_input_channels, n_output_channels,
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=True)

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def forward(self, x):
        return self.flattener(self.conv_layer(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            
class TransformerEncoder(Module):
    def __init__(self,
                 embedding_dim=256,
                 num_layers=6,
                 num_heads=4,
                 mlp_ratio=2,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 positional_embedding=False,
                 sequence_length=None,
                 cls_token = False):
        super().__init__()
        self.cls_token = cls_token
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim

        if cls_token:
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                        requires_grad=True)
        ### TODO
        if positional_embedding:
            assert sequence_length != None
            self.positional_emb = Parameter(self.sinusoidal_embedding(\
                              sequence_length, embedding_dim), requires_grad=False)
        ### ###
        
        self.dropout = Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.apply(self.init_weight)

    def forward(self, x):
        
        if self.cls_token:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = self.dropout(x)
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            features.append(self.norm(x))
        return torch.stack(features, -3)

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)
    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class Attention(Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
