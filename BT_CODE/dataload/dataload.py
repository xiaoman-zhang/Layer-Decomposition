from __future__ import print_function
import cv2
import os
import torch
import numpy as np
import pandas as pd
from scipy import ndimage
import elasticdeform

import random
from typing import Optional, List, Callable, Any
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from skimage.transform import resize
import SimpleITK as sitk
import torchio as tio

from utils.vtk_itk import pd_to_itk_image
from utils.prelin_sphere import *
from dataload.augmentations import * 


class Dataset_train(Dataset):
    def __init__(self,oasis_data_list,train_data_list,transform,crop_size):
        ployhedron_path = './resources/geodesic_polyhedron.vtp'
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(ployhedron_path)
        reader.Update()
        self.input_poly_data = reader.GetOutput()

        self.itk_img_SetOrigin = (0.0, 0.0, 0.0)

        self.normal_file_path = np.asarray(oasis_data_list.iloc[:,0])
        self.datafile_path = np.asarray(train_data_list.iloc[:,0])
        self.flair_file_path = np.asarray(train_data_list.iloc[:,1])

        self.gen_file_path = np.concatenate((self.flair_file_path ,self.normal_file_path))
        self.data_len_gen = len(self.gen_file_path)
        self.data_len_normal = len(self.normal_file_path)
        self.data_len = len(self.datafile_path)
        self.crop_size = crop_size

        # self.trans = tio.RandomBlur()
        self.inten = True
        self.trans =tio.Compose((
                tio.RandomElasticDeformation(num_control_points=7, locked_borders=2),
                tio.RandomBlur()))

    def __getitem__(self,index):
        normal_index = random.randint(0,self.data_len_normal-1)
        normal_img_path = self.normal_file_path[normal_index]
        itk_img_normal = sitk.ReadImage(normal_img_path)
        itk_img_normal.SetOrigin(self.itk_img_SetOrigin)
        normal_array = sitk.GetArrayFromImage(itk_img_normal)
        # normal_array = self.aug_sample(normal_array)
        normal_mean = np.mean(normal_array[np.nonzero(normal_array)])

        gen_tumor_index = random.randint(0,self.data_len_gen-1)
        gen_img_path = self.gen_file_path[gen_tumor_index]
        itk_img_gen = sitk.ReadImage(gen_img_path)
        itk_img_gen.SetOrigin(self.itk_img_SetOrigin)
        gen_array = sitk.GetArrayFromImage(itk_img_gen)

        tumor_mask = self.tumor_array_generate(gen_array,itk_img_gen,self.input_poly_data)
        tumor_gen = self.tumor_generate(gen_array,tumor_mask,normal_mean,self.trans)
        normal_crop,tumor_crop,mask_crop = self.random_crop(normal_array,gen_array,tumor_gen,tumor_mask)
        mask_crop[normal_crop==0]=0
        mix_crop,mix_tumor = self.mix_array(normal_crop,tumor_crop,mask_crop)

        normal_crop = self.normalize_mri(normal_crop)
        tumor_norm = self.normalize_mri(mix_tumor)
        mix_crop = self.normalize_mri(mix_crop)

        x_normal = normal_crop[np.newaxis,:,:,:]
        x_tumor = tumor_norm[np.newaxis,:,:,:]
        x_train = mix_crop[np.newaxis,:,:,:]
        x_mask = mask_crop[np.newaxis,:,:,:]

        return (torch.tensor(x_train.copy(), dtype=torch.float),
                torch.tensor(x_normal.copy(), dtype=torch.float),
                torch.tensor(x_tumor.copy(), dtype=torch.float),
                torch.tensor(x_mask.copy(), dtype=torch.float))
    

    def tumor_array_generate(self,img_array, itk_img, input_poly_data):
        brain_start_x,brain_end_x,brain_start_y,brain_end_y,brain_start_z,brain_end_z = self.crop_img(img_array)
        ratio_x = np.random.uniform(0.3,0.8)
        ratio_y = np.random.uniform(0.2,0.7)
        if np.random.random()< 0.5:
            ratio_z = np.random.uniform(0.24,0.44)
        else:
            ratio_z = np.random.uniform(0.56,0.76)
        
        tumor_center_x = int((brain_end_x-brain_start_x)*ratio_x + brain_start_x)
        tumor_center_y = int((brain_end_y-brain_start_y)*ratio_y + brain_start_y)
        tumor_center_z = int((brain_end_z-brain_start_z)*ratio_z + brain_start_z)
        center = [tumor_center_z,tumor_center_y,tumor_center_x]

        k = 4/3 * np.pi
        volume = np.random.randint(7000, 200000)
        radius = (volume / k) ** (1/3)
        ratio = np.random.uniform(0.8, 1)
        a = radius
        b = radius / ratio
        c = radius * ratio

        radii = c,b,a
        angles = np.random.uniform(0, 180, size=3)

        octaves = np.random.randint(4, 8)
        offset = np.random.randint(1000)
        scale = 0.5

        output_poly_data = get_resection_poly_data(input_poly_data,offset,center,radii,angles,octaves,scale)
        output_poly_stk = pd_to_itk_image(output_poly_data,itk_img)
        output_poly_array = sitk.GetArrayFromImage(output_poly_stk)
        output_poly_array[output_poly_array>0]=1
        return output_poly_array


    def crop_img(self,img_array):
        zx = np.any(img_array, axis=(1,2))
        start_slicex, end_slicex = np.where(zx)[0][[0, -1]]
        zy = np.any(img_array, axis=(0,2))
        start_slicey, end_slicey = np.where(zy)[0][[0, -1]]   
        zz = np.any(img_array, axis=(0,1))
        start_slicez, end_slicez = np.where(zz)[0][[0, -1]]
        return start_slicex, end_slicex, start_slicey, end_slicey, start_slicez, end_slicez
    
    def aug_sample(self,x):
        if random.random() < 0.5:
            x = np.flip(x, axis=0)
        if random.random() < 0.5:
            x = np.flip(x, axis=1)
        if random.random() < 0.5:
            x = np.flip(x, axis=2)
        return x
    
    def mix_array(self,normal,tumor,mask):
        mix_ratio_value = random.uniform(0.2,0.8)
        mix_mask = mask*mix_ratio_value
        mix_array = np.multiply(mix_mask, tumor) + np.multiply((1-mix_mask), normal)
        tumor[mask!=0]=mix_array[mask!=0]
        return mix_array,tumor
    
    def normalize_mri(self,npimg):
        image_nonzero = npimg[np.nonzero(npimg)]
        if np.mean(npimg)==0:
            print('tumor is 0!')
            tmp =  npimg
        elif np.std(image_nonzero) < 0.01: 
            print('nonzero std is 0!')
            tmp =  (npimg - np.min(npimg))/(np.max(npimg)-np.min(npimg))
        else: 
            tmp=(npimg - np.mean(image_nonzero))/ np.std(image_nonzero)
        return tmp

    def tumor_generate(self,img_array,tumor_mask,normal_mean,trans):
        tumor_gen = img_array.copy()[np.newaxis,:,:,:]
        tumor_aug = np.array(trans(tumor_gen))[0]
        tumor_aug[tumor_mask==0]=0
        tumor_mean = np.mean(tumor_aug[np.nonzero(tumor_aug)])
        if self.inten:
            intensity_index = random.uniform(1.0,3.0)
            tumor_aug = tumor_aug*(normal_mean*intensity_index/tumor_mean)
        else:
            tumor_aug  = tumor_aug
        return tumor_aug
    
    def random_crop(self, img, gen, tumor, mask):
        zx = np.any(img, axis=(1,2))
        start_slicex, end_slicex = np.where(zx)[0][[0, -1]]
        zy = np.any(img, axis=(0,2))
        start_slicey, end_slicey = np.where(zy)[0][[0, -1]]
        zz = np.any(img, axis=(0,1))
        start_slicez, end_slicez = np.where(zz)[0][[0, -1]]

        normal_img = img[start_slicex:end_slicex, start_slicey:end_slicey,start_slicez:end_slicez]
        
        zx_gen = np.any(gen, axis=(1,2))
        start_slicex_gen, end_slicex_gen = np.where(zx_gen)[0][[0, -1]]
        zy_gen = np.any(gen, axis=(0,2))
        start_slicey_gen, end_slicey_gen = np.where(zy_gen)[0][[0, -1]]
        zz_gen = np.any(gen, axis=(0,1))
        start_slicez_gen, end_slicez_gen = np.where(zz_gen)[0][[0, -1]]
        tumor_img = tumor[start_slicex_gen:end_slicex_gen, start_slicey_gen:end_slicey_gen,start_slicez_gen:end_slicez_gen]
        tumor_mask = mask[start_slicex_gen:end_slicex_gen, start_slicey_gen:end_slicey_gen,start_slicez_gen:end_slicez_gen]

        normal_img = resize(normal_img,self.crop_size,order=1,preserve_range=True)
        tumor_img = resize(tumor_img,self.crop_size,order=0,preserve_range=True)
        tumor_mask = resize(tumor_mask,self.crop_size,order=0,preserve_range=True)
        return normal_img.astype(np.int16), tumor_img.astype(np.int16), tumor_mask.astype(np.int16)

    def __len__(self):
        return len(self.datafile_path)
