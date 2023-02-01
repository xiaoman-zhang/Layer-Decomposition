import argparse
import os
import time
import logging
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data
from tensorboardX import SummaryWriter
from torchsummary import summary
from skimage.transform import resize
import torch.nn.functional as F
import SimpleITK as sitk


from model import unet3d
from utils.loss import DiceLoss
from dataload.dataload import *

parser = argparse.ArgumentParser(description='Layer Decomposition for 3D Tumor Segmentation')
#dataset path
#oasis_csv_dir: data from OASIS (used as normal data)
#train_csv_dir: data from BraTS (can be added after warming up/ used for tumor generater)
parser.add_argument('--oasis_csv_dir', default='data_csv/OASIS.csv',help='dataset path')
parser.add_argument('--train_csv_dir', default='data_csv/BraTS.csv',help='dataset path')
parser.add_argument('--test_csv_dir', default='../DATA/data_csv/brats',help='dataset path')
parser.add_argument('--fold', default=0, type=int,
                    metavar='fold', help='train fold')
parser.add_argument('--transformation', default=0, type=int,
                    metavar='N', help='transformation  (default: 2)')

parser.add_argument('--crop_size',default=[128,128,128])
parser.add_argument('--load'  , type=bool,  default=False,  help='whether loaded')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 2)')

parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--patience', default=40, type=int)
parser.add_argument('--patience_lr', default=10, type=int)


parser.add_argument('--model_path', type=str, default='model',
                    help='Name of Experiment')
parser.add_argument('--snapshot_path', type=str, default='train',help='Name of Experiment')
parser.add_argument('--model_name', type=str, default='model.pt',help='Name of Experiment')
parser.add_argument('--best_test_model_name', type=str, default='best_test_model.pt',help='Name of Experiment')
parser.add_argument('--best_valid_model_name', type=str, default='best_valid_model.pt',help='Name of Experiment')
parser.add_argument('--validation_path', type=str, default='model',
                    help='Name of Experiment')
parser.add_argument('--gpu', type=str,default='5', help='gpu') 
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.cuda.current_device()
torch.cuda._initialized = True

def train():
    oasis_data_list = pd.read_csv(args.oasis_csv_dir)
    train_data_list = pd.read_csv(args.train_csv_dir)

    train_dataset = Dataset_train(oasis_data_list,train_data_list,args.transformation,args.crop_size)
    train_loader =  torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count())

    model = unet3d.UNet3D(n_class=3, act='relu')
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=3e-5) 
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = args.patience_lr, mode = 'min')

    criterion = nn.L1Loss()
    criterion_ce = nn.BCELoss()
    criterion_dice = DiceLoss()

    if args.load:
        snapshot_path = os.path.join(args.model_path, args.snapshot_path)
        checkpoint = torch.load(os.path.join(snapshot_path,'model_state.pt'))
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)   
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        intial_epoch= checkpoint['epoch']
        print('load model from:',os.path.join(snapshot_path,args.model_name))
    else:
        intial_epoch=0
    train_model(intial_epoch,model,optimizer,scheduler,criterion,criterion_ce,criterion_dice,train_loader,device)

def train_model(intial_epoch,model,optimizer,scheduler,criterion,criterion_ce,criterion_dice,train_loader,device):
    snapshot_path = os.path.join(args.model_path, args.snapshot_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(train_loader)))
    logging.info(optimizer.state_dict()['param_groups'][0]['lr'])

    scalar_step = intial_epoch*len(train_loader)
    best_loss=10.0
    best_valid_loss = 10.0
    best_test_dice = 0.0

    num_epoch_no_improvement=0
    # to track the training loss as the model trains
    train_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    avg_valid_losses = []

    for epoch in range(intial_epoch,args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)
        model.train()
        for i, (input, target_normal,target_tumor,tumor_mask) in enumerate(train_loader):
            input_var = input.to(device)
            target_normal = target_normal.to(device)
            target_tumor = target_tumor.to(device)
            tumor_mask = tumor_mask.to(device)

            optimizer.zero_grad()
            output = model(input_var)
            #output three channel-normal/tumor/segmentation mask
            output_normal = output[:,0:1,:,:,:]
            output_tumor = output[:,1:2,:,:,:]
            output_mask = F.sigmoid(output[:,2:3,:,:,:])

            loss_normal_all = criterion(output_normal,target_normal)
            loss_tumor_all = criterion(output_tumor,target_tumor)
            loss_mask = criterion_dice(output_mask,tumor_mask)

            if len(torch.unique(tumor_mask)) == 2:
                #Emphasis on normal part & tumor part
                loss_normal = criterion(output_normal[tumor_mask!=0],target_normal[tumor_mask!=0])
                loss_tumor = criterion(output_tumor[tumor_mask!=0],target_tumor[tumor_mask!=0])
                loss = loss_mask + loss_normal_all + loss_tumor_all + loss_normal + loss_tumor 
                writer.add_scalar('loss/loss_normal', loss_normal, scalar_step)
                writer.add_scalar('loss/loss_tumor', loss_tumor, scalar_step)
                print('epoch %d iteration %d :  loss: %f, loss_normal_all: %f,loss_tumor_all: %f, loss_normal: %f,loss_tumor: %f,loss_mask: %f' % (epoch,scalar_step, loss.item(),loss_normal_all.item(),loss_tumor_all.item(), loss_normal.item(),loss_tumor.item(),loss_mask.item()))
            else:
                loss =  loss_mask + loss_normal_all + loss_tumor_all
                print('epoch %d iteration %d :  loss: %f, loss_normal_all: %f,loss_tumor_all: %f,loss_mask: %f' % (epoch,scalar_step, loss.item(),loss_normal_all.item(),loss_tumor_all.item(),loss_mask.item()))
            
            loss.backward()
            optimizer.step()
            loss = loss.float()
            train_losses.append(round(loss.item(), 2))
            writer.add_scalar('loss/train_loss', loss, scalar_step)
            writer.add_scalar('loss/loss_normal_all', loss_normal_all, scalar_step)
            writer.add_scalar('loss/loss_tumor_all', loss_tumor_all, scalar_step)
            writer.add_scalar('loss/loss_mask', loss_mask, scalar_step)
            scalar_step += 1
            train_loss=np.average(train_losses)
        writer.add_scalar('loss/train_loss_epoch', train_loss, epoch)
        avg_train_losses.append(train_loss)
        scheduler.step(train_loss)
        test_dice, test_dice_mask, valid_loss = test(epoch,model,criterion,writer,device,gate=0.5)
        writer.add_scalar('loss/test_dice', test_dice, epoch)
        writer.add_scalar('loss/test_dice_mask', test_dice_mask, epoch)
        writer.add_scalar('loss/valid_loss', valid_loss, epoch)
        logging.info("Epoch {}, test_dice is {:.4f},test_dice_mask is {:.4f}, valid loss is {:.4f}".format(epoch+1,test_dice,test_dice_mask,valid_loss))
        train_losses=[]

        if epoch < 30 and valid_loss < best_valid_loss:
            logging.info("Epoch {}, saving best valid model, test_dice is {:.4f},test_dice_mask is {:.4f}, valid loss is {:.4f}".format(epoch+1,test_dice,test_dice_mask, valid_loss))
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch+1,
                'state_dict' : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },os.path.join(snapshot_path,args.best_valid_model_name))
            print("Saving model ",os.path.join(snapshot_path,args.best_valid_model_name))
        elif valid_loss < best_valid_loss:
            logging.info("Epoch {}, saving best valid model, test_dice is {:.4f},test_dice_mask is {:.4f}, valid loss is {:.4f}".format(epoch+1,test_dice,test_dice_mask, valid_loss))
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch+1,
                'state_dict' : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },os.path.join(snapshot_path,'model_valid.pt'))
            print("Saving model ",os.path.join(snapshot_path,'model_valid.pt'))

        
        if test_dice_mask > best_test_dice:
            logging.info("Epoch {}, saving best test model, test_dice is {:.4f},test_dice_mask is {:.4f}, valid loss is {:.4f}".format(epoch+1,test_dice,test_dice_mask, valid_loss))
            best_test_dice = test_dice_mask
            torch.save({
                'epoch': epoch+1,
                'state_dict' : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },os.path.join(snapshot_path,args.best_test_model_name))
            print("Saving model ",os.path.join(snapshot_path,args.best_test_model_name))
        
        if train_loss < best_loss:
            logging.info("Epoch {}, Training loss decreases from {:.4f} to {:.4f}".format(epoch+1,best_loss, train_loss))
            best_loss = train_loss
            num_epoch_no_improvement = 0
            #save model
            torch.save({
                'epoch': epoch+1,
                'state_dict' : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },os.path.join(snapshot_path,args.model_name))
            print("Saving model ",os.path.join(snapshot_path,args.model_name))
        else:
            logging.info("Epoch {}, Training loss is {:.4f}, does not decrease from {:.4f}, num_epoch_no_improvement {}".format(epoch+1, train_loss,best_loss,num_epoch_no_improvement))
            num_epoch_no_improvement += 1
            logging.info(optimizer.state_dict()['param_groups'][0]['lr'])
            torch.save({
                'epoch': epoch+1,
                'state_dict' : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },os.path.join(snapshot_path,'model_state.pt'))
        
        if num_epoch_no_improvement >= args.patience or optimizer.state_dict()['param_groups'][0]['lr'] < 1e-6:
            logging.info("Early Stopping")
            torch.save({
                'epoch': epoch+1,
                'state_dict' : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },os.path.join(snapshot_path,'model_final.pt'))
            print("Saving model ",os.path.join(snapshot_path,'model_final.pt'))
            break
    torch.save({
                'epoch': epoch+1,
                'state_dict' : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },os.path.join(snapshot_path,'model_final.pt'))
    print("Saving model ",os.path.join(snapshot_path,'model_final.pt'))


def test(epoch,model,criterion,writer,device,gate):
    #three fold cross-validation
    test_csv_dir = os.path.join(args.test_csv_dir,'test_'+str(args.fold)+'.csv')
    test_data_list = pd.read_csv(test_csv_dir)
    flair_file_path = np.asarray(test_data_list.iloc[:,1])
    seg_file_path = np.asarray(test_data_list.iloc[:,5])
    scalar_step_test = epoch*len(test_data_list)

    model.eval()
    dice_sum = []
    mask_dice_sum = []
    valid_losses = []

    for index in range(len(flair_file_path)):
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
            output_normal_valid = out_patch[:,0:1,:,:,:]
            output_tumor_valid  = out_patch[:,1:2,:,:,:]
            output_mask = F.sigmoid(out_patch[:,2:3,:,:,:])
            output_input_valid = torch.mul(output_tumor_valid,output_mask) + torch.mul(output_normal_valid,(1-output_mask))
            valid_loss = criterion(output_input_valid,input_patch)
            writer.add_scalar('valid_loss/valid_loss', valid_loss, scalar_step_test)
            valid_loss = valid_loss.float()
            valid_losses.append(round(valid_loss.item(), 2))
            output_normal = out_patch[0,0,:,:,:].cpu().detach().numpy()
            output_tumor = F.sigmoid(out_patch[0,1,:,:,:]).cpu().detach().numpy()
            output_mask = output_mask[0][0].cpu().detach().numpy()
        output_tumor[output_tumor>gate]=1
        output_tumor[output_tumor<gate]=0
        output_mask[output_mask>gate]=1
        output_mask[output_mask<gate]=0
        tumor_seg = reshape_tumor(output_tumor,crop_axis,resize_shape,img_shape)
        mask_seg = reshape_tumor(output_mask,crop_axis,resize_shape,img_shape)
        dice_index = dice_coef(tumor_seg,seg_array)
        mask_dice_index = dice_coef(mask_seg,seg_array)
        dice_sum.append(dice_index)
        mask_dice_sum.append(mask_dice_index)
    mean_dice = np.mean(np.array(dice_sum))
    mean_dice_mask = np.mean(np.array(mask_dice_sum))
    mean_valid_loss = np.average(valid_losses)
    return mean_dice, mean_dice_mask,mean_valid_loss
  
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

if __name__ == '__main__':
    train()