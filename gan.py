# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 12:42:51 2018

@author: prodi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Generator G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, c=128):
        super(generator, self).__init__()
        self.Deconv_1 = nn.ConvTranspose2d(100, c*8, 4, 1, 0)
        self.Deconv_bn_1 = nn.BatchNorm2d(c*8)
        self.Deconv_2 = nn.ConvTranspose2d(c*8, c*4, 4, 2, 1)
        self.Deconv_bn_2 = nn.BatchNorm2d(c*4)
        self.Deconv_3 = nn.ConvTranspose2d(c*4, c*2, 4, 2, 1)
        self.Deconv_bn_3 = nn.BatchNorm2d(c*2)
        self.Deconv_4 = nn.ConvTranspose2d(c*2, c, 4, 2, 1)
        self.Deconv_bn_4 = nn.BatchNorm2d(c)
        self.Deconv_5 = nn.ConvTranspose2d(c, 3, 4, 2, 1)
    
    # weight initialization
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def forward(self, input):
        
        x = self.Deconv_1(input)
        x = self.Deconv_bn_1(x)
        x = F.relu(x)
        
        x = self.Deconv_2(x)
        x = self.Deconv_bn_2(x)
        x = F.relu(x)
        
        x = self.Deconv_3(x)
        x = self.Deconv_bn_3(x)
        x = F.relu(x)
        
        x = self.Deconv_4(x)
        x = self.Deconv_bn_4(x)
        x = F.relu(x)
        
        x = self.Deconv_5(x)
        
        return x
        

# Discriminator
class discriminator(nn.Module):
    # initializers
    def __init__(self, c=128):
        super(discriminator, self).__init__()
        self.Conv_1 = nn.Conv2d(3, c, 4, 2, 1)
        self.Conv_2 = nn.Conv2d(c, c*2, 4, 2, 1)
        self.Conv_bn_2 = nn.BatchNorm2d(c*2)
        self.Conv_3 = nn.Conv2d(c*2, c*4, 4, 2, 1)
        self.Conv_bn_3 = nn.BatchNorm2d(c*4)
        self.Conv_4 = nn.Conv2d(c*4, c*8, 4, 2, 1)
        self.Conv_bn_4 = nn.BatchNorm2d(c*8)
        self.Conv_5 = nn.Conv2d(c*8, 1, 4, 1, 0)
        
    # weight initialization
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def forward(self, input):
        
        x = self.Conv_1(input)
        x = F.leaky_relu(x, 0.2)
        
        x = self.Conv_2(x)
        x = self.Conv_bn_2(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.Conv_3(x)
        x = self.Conv_bn_3(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.Conv_4(x)
        x = self.Conv_bn_4(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.Conv_5(x)
        x = F.sigmoid(x)
        
        return x
    

#Normal initialization
def normal_init(m, mean, std):
    
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        
#Binary Cross Entropy loss function
def loss_fn(outputs, labels):
    
    BCE_loss = nn.BCELoss()
    
    return BCE_loss(outputs, labels)