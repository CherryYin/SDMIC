# --------------------------------------------------------
# SEMANTIC-MIL-NLP
# Copyright (c) 2021 Ubisoft Chengdu Studio
# Licensed under The MIT License [see LICENSE for details]
# Written by Yin Juan
# Refrence from Zhekun Luo
# --------------------------------------------------------

"""The task is predict is a comment some field relevant, 
   so input is text, output is a binary label"""

from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import fasttext
import numpy as np
import io
from torch.utils.tensorboard import SummaryWriter

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([float(v) for v in tokens[1:]])
    return data

class Semantic_MIL(nn.Module):
    def __init__(self, device, 
                 vocab_dict, 
                 vocab_num, 
                 emb_size, 
                 emb_path, 
                 max_len=256, 
                 n_hidden=512, 
                 gamma=0.4,
                 beta=0.7,
                 phase="train"):
        super(Semantic_MIL, self).__init__()
        self.device = device
        self.emb_size = 300  # The dimonsion of latent variable
        self.seq_len = 100
        self._gamma = gamma  # Classification threshold
        self._beta = beta
        self.kernel_sizes= [1, 2, 3]
        self.kernel_num = 256
        self.BCE = torch.nn.BCELoss()
        self.MSE = torch.nn.MSELoss(reduction='mean')
        self.KLD = torch.nn.KLDivLoss(reduction='mean')
        self.SL1 = torch.nn.SmoothL1Loss(reduce=False, size_average=False)
        if phase == "train":
            print("load embedding vectors...")
            org_emb_model = load_vectors(emb_path)
            emb_weight = np.random.random((vocab_num+2, emb_size))
            for w, idx in vocab_dict.items():
                if w in org_emb_model:
                    emb_weight[idx] = org_emb_model[w]
            emb_weight = torch.FloatTensor(emb_weight)
            self.embedding = nn.Embedding.from_pretrained(emb_weight, freeze=False)
            del emb_weight, org_emb_model
        else:
            self.embedding = nn.Embedding(vocab_num+2, emb_size)

        # key instance assignment
        yi_layers = []
        for kernel_size in self.kernel_sizes:
            yi_layers.append([nn.Conv2d(1, self.kernel_num, kernel_size=(kernel_size, self.emb_size), stride=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d((self.seq_len - kernel_size + 1, 1))])
        self.yi1 = nn.Sequential(*(yi_layers[0]))
        self.yi2 = nn.Sequential(*(yi_layers[1]))
        self.yi3 = nn.Sequential(*(yi_layers[2]))
       
        self.yi_cls = nn.Sequential(
            nn.Linear(self.kernel_num*len(self.kernel_sizes), 1),
            nn.Sigmoid()
        )
        
        # key instance assignment
        z_layers = []
        for kernel_size in self.kernel_sizes:
            z_layers.append([nn.Conv2d(1, self.kernel_num, kernel_size=(kernel_size, self.emb_size), stride=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d((self.seq_len - kernel_size + 1, 1))])
        self.z1 = nn.Sequential(*(z_layers[0]))
        self.z2 = nn.Sequential(*(z_layers[1]))
        self.z3 = nn.Sequential(*(z_layers[2]))
       
        self.z_cls = nn.Sequential(
            nn.Linear(self.kernel_num*len(self.kernel_sizes), 1),
            nn.Sigmoid()
        )

    def forward_base(self, comment_ids, keywords):
        # gated_machianism
        topic_ebms = self.embedding(keywords)
        topic_ebms = torch.mean(topic_ebms, axis=0)
        comment_emb = self.embedding(comment_ids)
        
        emb_prod = topic_ebms * comment_emb
        emb_prod = emb_prod.unsqueeze(1)
        yi_o1 = self.yi1(emb_prod)
        yi_o1 = yi_o1.squeeze()
        yi_o2 = self.yi2(emb_prod) #.permute(0, 2, 1)
        yi_o2 = yi_o2.squeeze()
        yi_o3 = self.yi3(emb_prod) #.permute(0, 2, 1)
        yi_o3 = yi_o3.squeeze()
        # print(k_comment1.size(), k_comment2.size(), k_comment3.size())
        yi = torch.cat([yi_o1, yi_o2, yi_o3], 1)
        # (50, 256*3)
        yi = self.yi_cls(yi)
        
        z_o1 = self.z1(emb_prod)
        z_o1 = z_o1.squeeze()
        z_o2 = self.z2(emb_prod) #.permute(0, 2, 1)
        z_o2 = z_o2.squeeze()
        z_o3 = self.z3(emb_prod) #.permute(0, 2, 1)
        z_o3 = z_o3.squeeze()
        # print(k_comment1.size(), k_comment2.size(), k_comment3.size())
        z = torch.cat([z_o1, z_o2, z_o3], 1)
        # (50, 256*3)
        z = self.z_cls(z)
        
        return yi, z
        
    def forward_M(self, comment_ids, label, stage, keywords):
        T = comment_ids.shape[0]   # The number of instances in a bag
        # k_comment, c_comment = self.forward_body(comment)
        yi, z = self.forward_base(comment_ids, keywords)
        z_pse = z[:, 0]
        pseudo_label = torch.zeros(T).to(self.device)

        if label == 1:
            threshold = torch.mean(z_pse) * self._beta
            pseudo_i = (z_pse > threshold).clone().detach().type(
                    'torch.FloatTensor').to(self.device)
            pseudo_label = torch.max(pseudo_label, pseudo_i)
        loss = self.BCE(yi.view(-1), pseudo_label)
        
        return loss, yi, z
    
    def forward_E(self, comment_ids, label, stage, keywords):
        T = comment_ids.shape[0]
        # k_comment, c_comment = self.forward_body(comment, keywords)
        yi, z = self.forward_base(comment_ids, keywords)

        pseudo_label = torch.zeros(T).to(self.device)

        yi_pse = yi[:, 0]
        z_log = torch.log(z)
        yi_pse = yi_pse.clone().detach()
        if label == 1:
            pseudo_label = (yi_pse-torch.min(yi_pse))/(torch.max(yi_pse) - torch.min(yi_pse))
            
        # z_avg = torch.mean(z)

        loss = self.KLD(z_log.view(-1), pseudo_label)
        # loss += self.BCE(c_avg, torch.FloatTensor(label))

        return loss, yi, z
    
    def forward_pred(self, comment):
        yi, z = self.forward_base(comment)
        yi_score = F.sigmoid(yi)
        z = F.sigmoid(z)
        
        return yi, z
    
    
def generate_model(vocab_dict, emb_path, gamma):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Semantic_MIL(device=device, 
                         vocab_dict=vocab_dict, 
                         vocab_num=len(vocab_dict), 
                         emb_size=300, 
                         emb_path=emb_path,
                         gamma=gamma).to(device)
    torch.backends.cudnn.enabled = False
    return model, model.parameters()
        
