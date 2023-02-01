import argparse
import os
import time
import logging
import sys
import math
import csv
import numpy as np
import pandas as pd

import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data
from tensorboardX import SummaryWriter
from torchsummary import summary
from skimage.transform import resize

from utils.metrics import calc_all_metrics,calculate_metric_percase
from model import unet3d


parser = argparse.ArgumentParser(description='evaluate model path')
#dataset path
parser.add_argument('--train_csv_dir', default='./data_csv/BraTS.csv',help='dataset path')
parser.add_argument('--test_csv_dir', default='./data_csv/brats',help='dataset path')
parser.add_argument('--crop_size',default=[128,128,128])
parser.add_argument('--fold', default=0, type=int)

parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 2)')

parser.add_argument('--model_save_path', type=str, default='../model/best_valid_model.pt',
                    help='Name of Experiment')
parser.add_argument('--result_save_path', type=str, default='../result',
                    help='Name of Experiment')
parser.add_argument('--snapshot_path', type=str, default='fold_0',
                    help='Name of Experiment')
parser.add_argument('--gpu', type=str,default='0', help='gpu') 
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.cuda.current_device()
torch.cuda._initialized = True

def normalize_mri(npimg):
    image_nonzero = npimg[np.nonzero(npimg)]
    tmp=(npimg - np.mean(image_nonzero))/ np.std(image_nonzero)
    return tmp


def dice_coef(input,target):
    """soft dice loss"""
    eps = 1e-7
    intersection = np.multiply(input,target).sum()
    return  (2. * intersection+eps) / (input.sum() +target.sum() + eps)

def reshape_tumor(img,crop_axis,crop_shape,img_shape):
    normal_img = resize(img,crop_shape,order=0,preserve_range=True)
    start_slicex,end_slicex,start_slicey,end_slicey,start_slicez,end_slicez = crop_axis
    output_img = np.zeros(img_shape)
    output_img[start_slicex:end_slicex, start_slicey:end_slicey,start_slicez:end_slicez] = normal_img
    return output_img

def crop_brain(img,seg):
    #  crop_brain
    zx = np.any(img, axis=(1,2))
    start_slicex, end_slicex = np.where(zx)[0][[0, -1]]
    zy = np.any(img, axis=(0,2))
    start_slicey, end_slicey = np.where(zy)[0][[0, -1]]
    zz = np.any(img, axis=(0,1))
    start_slicez, end_slicez = np.where(zz)[0][[0, -1]]
    normal_img = img[start_slicex:end_slicex, start_slicey:end_slicey,start_slicez:end_slicez]
    crop_seg = seg[start_slicex:end_slicex, start_slicey:end_slicey,start_slicez:end_slicez]
    normal_img = resize(normal_img,args.crop_size,order=1,preserve_range=True)
    crop_seg = resize(crop_seg,args.crop_size,order=0,preserve_range=True)
    return normal_img,crop_seg

def crop_brain_dice(img):
    zx = np.any(img, axis=(1,2))
    start_slicex, end_slicex = np.where(zx)[0][[0, -1]]
    zy = np.any(img, axis=(0,2))
    start_slicey, end_slicey = np.where(zy)[0][[0, -1]]
    zz = np.any(img, axis=(0,1))
    start_slicez, end_slicez = np.where(zz)[0][[0, -1]]
    normal_img = img[start_slicex:end_slicex, start_slicey:end_slicey,start_slicez:end_slicez]
    crop_axis = [start_slicex,end_slicex,start_slicey,end_slicey,start_slicez,end_slicez]
    crop_shape = normal_img.shape
    normal_img = resize(normal_img,args.crop_size,order=1,preserve_range=True)
    return normal_img,crop_axis,crop_shape

def test():
    if not os.path.exists(args.result_save_path):
        os.makedirs(args.result_save_path)
    save_result_path = os.path.join(args.result_save_path,args.snapshot_path)
    if not os.path.exists(save_result_path):
        os.makedirs(save_result_path)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 

    test_dir = os.path.join(args.test_csv_dir,'test_'+str(args.fold)+'.csv')
    test_data_list = pd.read_csv(test_dir)
    flair_file_path = np.asarray(test_data_list.iloc[:,1])
    seg_file_path = np.asarray(test_data_list.iloc[:,5])

    model = unet3d.UNet3D(n_class=3, act='relu')
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model.to(device)

    checkpoint = torch.load(args.model_save_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)  
    print('load successful') 

    for index in range(len(test_data_list)):
        img_path = flair_file_path[index]
        itk_img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(itk_img)
        seg_path = seg_file_path[index]
        itk_img_seg = sitk.ReadImage(seg_path)
        seg_array = sitk.GetArrayFromImage(itk_img_seg)

        img_brain,img_seg = crop_brain(img_array,seg_array)
        img_brain_norm = normalize_mri(img_brain)
        img_array = img_brain_norm[np.newaxis,np.newaxis,:,:,:]
        input_patch = torch.tensor(img_array.copy(), dtype=torch.float).to(device)
        with torch.no_grad():
            model.eval()
            out_patch = model(input_patch)
            output_normal = F.sigmoid(out_patch[0,0,:,:,:]).cpu().detach().numpy()
            output_tumor = F.sigmoid(out_patch[0,1,:,:,:]).cpu().detach().numpy()
            mask_tumor = F.sigmoid(out_patch[0,2,:,:,:]).cpu().detach().numpy()
        output_save = sitk.GetImageFromArray((output_normal*255).astype(np.int16))
        sitk.WriteImage(output_save,os.path.join(save_result_path,'output_normal_'+str(index)+'.nii.gz'))
        output_save = sitk.GetImageFromArray((output_tumor*255).astype(np.int16))
        sitk.WriteImage(output_save,os.path.join(save_result_path,'output_tumor_'+str(index)+'.nii.gz'))
        output_save = sitk.GetImageFromArray((mask_tumor*255).astype(np.int16))
        sitk.WriteImage(output_save,os.path.join(save_result_path,'output_mask_'+str(index)+'.nii.gz'))
        output_save = sitk.GetImageFromArray((img_brain).astype(np.int16))
        sitk.WriteImage(output_save,os.path.join(save_result_path,'input_'+str(index)+'.nii.gz'))
        output_save = sitk.GetImageFromArray((img_seg).astype(np.int16))
        sitk.WriteImage(output_save,os.path.join(save_result_path,'seg_'+str(index)+'.nii.gz'))
        print(index)


def dice_cacluate(gate):
    if not os.path.exists(args.result_save_path):
        os.makedirs(args.result_save_path)
    save_result_path = args.result_save_path
    csv_path = os.path.join(save_result_path,str(args.fold)+'_metric.csv')
    dist_csv_col = [
    'Index',
    'Dice',
    'Jaccard',
    'Hausdorff',
    'ASD',
    'Sensitivity',
    'Specificity'
    ]
    f_train = open(csv_path,'w+',newline='')
    wf_train = csv.writer(f_train)
    wf_train.writerow(dist_csv_col)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 

    test_dir = os.path.join(args.test_csv_dir,'test_'+str(args.fold)+'.csv')
    test_data_list = pd.read_csv(test_dir)
    flair_file_path = np.asarray(test_data_list.iloc[:,1])
    seg_file_path = np.asarray(test_data_list.iloc[:,5])

    model = unet3d.UNet3D(n_class=3, act='relu')
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model.to(device)

    checkpoint = torch.load(args.model_save_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)  
    print('load successful') 

    dice_sum = []
    JC_sum = []
    HD_sum = []
    ASD_sum = []
    SEN_sum = []
    SPEC_sum = []

    for index in range(len(test_data_list)):
        img_path = flair_file_path[index]
        itk_img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(itk_img)

        seg_path = seg_file_path[index]
        itk_img_seg = sitk.ReadImage(seg_path)
        seg_array = sitk.GetArrayFromImage(itk_img_seg)
        seg_array[seg_array>0]=1
        img_shape = seg_array.shape

        img_brain,crop_axis,resize_shape = crop_brain_dice(img_array)
        img_brain_norm = normalize_mri(img_brain)
        img_array = img_brain_norm[np.newaxis,np.newaxis,:,:,:]
        input_patch = torch.tensor(img_array.copy(), dtype=torch.float).to(device)
        with torch.no_grad():
            model.eval()
            out_patch = model(input_patch)
            output_normal = out_patch[0,0,:,:,:].cpu().detach().numpy()
            output_tumor = F.sigmoid(out_patch[0,1,:,:,:]).cpu().detach().numpy()
            mask_tumor = F.sigmoid(out_patch[0,2,:,:,:]).cpu().detach().numpy()
        output_tumor[output_tumor>gate]=1
        output_tumor[output_tumor<gate]=0
        tumor_seg = reshape_tumor(output_tumor,crop_axis,resize_shape,img_shape)

        dice_index = dice_coef(tumor_seg,seg_array)
        dice, jc, hd, asd, sen, spec = calculate_metric_percase(tumor_seg,seg_array)
        dice_sum.append(dice_index) 
        JC_sum.append(jc)
        HD_sum.append(hd)
        ASD_sum.append(asd)
        SEN_sum.append(sen)
        SPEC_sum.append(spec)

        result_wirte = []
        result_wirte.append(index)
        result_wirte.append(dice)
        result_wirte.append(jc)
        result_wirte.append(hd)
        result_wirte.append(asd)
        result_wirte.append(sen)
        result_wirte.append(spec)
        wf_train.writerow(result_wirte)
        print(index,dice_index,result_wirte)
    mean_dice = np.mean(np.array(dice_sum))
    mean_JC = np.mean(np.array(JC_sum))
    mean_HD = np.mean(np.array(HD_sum))
    mean_ASD = np.mean(np.array(ASD_sum))
    mean_SEN = np.mean(np.array(SEN_sum))
    mean_SPEC = np.mean(np.array(SPEC_sum))

    result_wirte = []
    result_wirte.append('mean')
    result_wirte.append(mean_dice)
    result_wirte.append(mean_JC)
    result_wirte.append(mean_HD)
    result_wirte.append(mean_ASD)
    result_wirte.append(mean_SEN)
    result_wirte.append(mean_SPEC)
    wf_train.writerow(result_wirte)
    f_train.close()

if __name__ == '__main__':
    test()
    dice_cacluate(gate=0.5)