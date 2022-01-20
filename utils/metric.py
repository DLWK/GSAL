import glob
import numpy as np
import torch
import os
import cv2
import csv
from medpy import metric
def dice_sorce(y_true,y_pred):

    smooth = 1.
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true + y_pred)
    numerator = 2.0 * intersection+smooth
    denominator = union+smooth
    coef = numerator / denominator
    return coef

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    ious = (intersection + smooth) / (union + smooth)
    return  ious

def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()
    senstiv =  (intersection + smooth) / (target.sum() + smooth)

    return  senstiv

def ppv(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    ppvs= (intersection + smooth) / (output.sum() + smooth)
    return ppvs

def get_DC(SR,GT,threshold=0.6):
    
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC