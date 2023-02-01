#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Dice_Coef(nn.Module):
    def __init__(self):
        super(Dice_Coef, self).__init__()
     
    def forward(self, predict, target,weights=False):
        smooth = 1e-5 
        total_dice=0
        weight_sum = 0
        target = make_one_hot_3d(target,2)
        predict =  make_one_hot_3d(predict,2)
        for i in range(target.shape[0]):
            dice = dice_coef_binary_np(predict[i], target[i])
            total_dice += dice
        return total_dice/target.shape[0]

def dice_coef(input, target):
    """soft dice loss"""
    eps = 1e-7
    intersection = np.multiply(input,target).sum()
    return  (2. * intersection+eps) / (input.sum() +target.sum() + eps)

def make_one_hot_3d(npmask, num_classes):
    w,h,d = npmask.shape
    result = np.zeros((num_classes,w,h,d))
    result[0]=1-npmask
    result[1]=npmask
    # result = result.scatter_(1, npmask, 1)
    return result

def make_one_hot(npmask, num_classes):
    """
    Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(npmask.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, npmask.cpu(), 1).cuda()
    return result

def dice_coef_binary(predict, target):
    """soft dice loss"""
    eps = 1e-7
    intersection = torch.sum(torch.mul(predict, target))
    return  (2. * intersection+eps) / (torch.sum(predict) +torch.sum(target) + eps)

def dice_coef_binary_np(input, target):
    """soft dice loss"""
    eps = 1e-7
    intersection = np.multiply(input,target).sum()
    return  (2. * intersection+eps) / (input.sum() +target.sum() + eps)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
     
    def forward(self, predict, target,weights=False):
        smooth = 1e-5
        loss_record=[]
        total_loss=0
        weight_sum = 0
        dice_loss = 1-dice_coef_binary(predict[:,0],target[:, 0])
        return dice_loss
