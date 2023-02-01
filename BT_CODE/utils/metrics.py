#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage as ndi

from medpy import metric

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    sen = metric.sensitivity(pred, gt)
    spec = metric.specificity(pred, gt)
    return dice, jc, hd, asd, sen, spec

def mc_calc_localized_metrics(y_true, y_pred):
  """
  Calculate localized metrics. We care about accuracy near the knee joint
  more than along the femur/tibia bone shaft. Splits the true/predicted mask
  below the centroid and calculate metrics only for that region.
  
  Assumption: Label 1 = femur and 2 = tibia
  """
  dice_scores = [-1] # Keep placeholder for background score for compatibility
  localized_pred_masks = np.zeros_like(y_pred)
  for curr_class in range(1, 3):
    true_mask = np.zeros_like(y_true)
    pred_mask = np.zeros_like(y_pred)
    true_mask[np.where(y_true == curr_class)] = 1
    pred_mask[np.where(y_pred == curr_class)] = 1
    
    # Calculate centroid of ground truth mask
    centroid = ndi.center_of_mass(true_mask)
    centroid_y = centroid[1]
    
    # Keep desired volume (below center of mass)
    if curr_class == 1: # Femur keep below centroid
      true_mask[:,:int(centroid_y),:] = 0
      pred_mask[:,:int(centroid_y),:] = 0
    else: # Tibia keep above centroid
      true_mask[:,int(centroid_y):,:] = 0
      pred_mask[:,int(centroid_y):,:] = 0
    
    dice_score = dice_coef(true_mask, pred_mask)
    dice_scores.append(dice_score)
    # Return localized predicted masks as well
    localized_pred_masks[np.where(pred_mask == 1)] = curr_class
    
  return (dice_scores, localized_pred_masks)
    
def calc_volumetric_metrics(y_true, y_pred):
  """
  Convenience function to calculate only volumetric metrics which are faster
  to calculate and return as a list.
  Calculate DICE, VOE, and VD as percentages.
  """
  
  dice = dice_coef(y_true, y_pred)*100.
  VOE  = voe_vol_overlap_err(y_true, y_pred)*100.
  VD   = vd_rel_vol_diff(y_true, y_pred)*100.
  
  metrics = [dice,VOE,VD]
  return metrics
  
def calc_all_metrics(y_true, y_pred, voxel_spacing=None):
  """
  Convenience function to calculate all metrics and return as a list.
  DICE, VOE, and VD are returned as percentage.
  """
  
  dice = dice_coef(y_true, y_pred)*100.
  VOE  = voe_vol_overlap_err(y_true, y_pred)*100.
  VD   = vd_rel_vol_diff(y_true, y_pred)*100.
  
  sd1 = get_surf_dists(y_pred, y_true, voxel_spacing)
  sd2 = get_surf_dists(y_true, y_pred, voxel_spacing)
  
  n_pred = sd1.shape[0]
  n_true = sd2.shape[0]
  
  ASD = (np.sum(sd1) + np.sum(sd2)) / (n_true + n_pred)
  RSD = np.sqrt( (np.sum(sd1**2) + np.sum(sd2**2)) / (n_true + n_pred) )
  MSD = max(sd1.max(), sd2.max())
  
  metrics = [dice,VOE,VD,ASD,RSD,MSD]
  return metrics
  
def dice_coef(y_true, y_pred):
  """
  Calculate DICE coefficient score for predicted mask. Expects true and
  predicted masks to be binary arrays [0/1].
  
  # Arguments:
      y_true: numpy array of true targets, y_true.shape = [slices, H, W]
      y_pred: numpy array of predicted targets, y_pred.shape = [slices, H, W]
  # Returns:
      Scalar DICE coefficient for each label in the range [0 1]
  """
  intersection = np.sum(y_true * y_pred)
  union = np.sum(y_true) + np.sum(y_pred)
    
  if union == 0:
    union = 1
  dice_score = (2*intersection)/union
  return dice_score

def dice_coef_mc(y_true, y_pred, nb_classes):
  """
  Calculate DICE coefficient score for each class separately in case of multi
  class segmentation.
  
  # Arguments:
      y_true: numpy array of true targets, y_true.shape = [slices, H, W]
      y_pred: numpy array of predicted targets, y_pred.shape = [slices, H, W]
      nb_classes: number of different labels in true/predicted targets
  # Returns:
      list of DICE coefficients for each label in the range [0 1]
  """
  dice_scores = []
  for curr_class in range(nb_classes):
    true_mask = np.zeros_like(y_true)
    pred_mask = np.zeros_like(y_pred)
    true_mask[np.where(y_true == curr_class)] = 1
    pred_mask[np.where(y_pred == curr_class)] = 1
    
    intersection = np.sum(true_mask * pred_mask)
    union = np.sum(true_mask) + np.sum(pred_mask)
    
    if union == 0:
      union = 1
    dice_score = (2*intersection)/union
    dice_scores.append(dice_score)
    
  return dice_scores

def voe_vol_overlap_err(y_true, y_pred):
  """
  Calculate Volume Overlap Error [VOE] for predicted mask. Expects true and
  predicted masks to be binary arrays [0/1].
                DICE
  VOE = 1 -  -----------
              2 - DICE
  # Arguments:
      y_true: numpy array of true targets, y_true.shape = [slices, H, W]
      y_pred: numpy array of predicted targets, y_pred.shape = [slices, H, W]
  # Returns:
      Scalar VOE in the range [0 1]
  """
  dsc = dice_coef(y_true, y_pred)
  
  return 1. - dsc/(2. - dsc)

def vd_rel_vol_diff(y_true, y_pred):
  """
  Calculate Relative Volume Difference [RVD] for predicted mask. Expects true and
  predicted masks to be binary arrays [0/1].
         |y_pred| - |y_true|
  RVD = ----------------------
              |y_true|
  # Arguments:
      y_true: numpy array of true targets, y_true.shape = [slices, H, W]
      y_pred: numpy array of predicted targets, y_pred.shape = [slices, H, W]
  # Returns:
      Scalar RVD in the range [0 1]
  """
  
  nb_true = np.float64(np.sum(y_true))
  nb_pred = np.float64(np.sum(y_pred))
  
  if nb_true < 1:
    return 0
  
  return (nb_pred - nb_true) / nb_true

def asd_avg_surf_dist(y_true, y_pred, voxel_spacing=None):
  """
  Calculate Average Surface Distance [ASD] for predicted mask. Expects true and
  predicted masks to be binary arrays [0/1].
  
  # Arguments:
      y_true: numpy array of true targets, y_true.shape = [slices, H, W]
      y_pred: numpy array of predicted targets, y_pred.shape = [slices, H, W]
      voxel_spacing: tuple with voxel spacing in 3 dimensions
  # Returns:
      Scalar ASD in mm
  """

  sd1 = get_surf_dists(y_pred, y_true, voxel_spacing)
  sd2 = get_surf_dists(y_true, y_pred, voxel_spacing)
  
  n_pred = sd1.shape[0]
  n_true = sd2.shape[0]
  
  asd = (np.sum(sd1) + np.sum(sd2)) / (n_true + n_pred)
  
  return asd

def rsd_rms_surf_dist(y_true, y_pred, voxel_spacing=None):
  """
  Calculate Root Mean Square Surface Distance [RSD] for predicted mask.
  Expects true and predicted masks to be binary arrays [0/1].
  
  # Arguments:
      y_true: numpy array of true targets, y_true.shape = [slices, H, W]
      y_pred: numpy array of predicted targets, y_pred.shape = [slices, H, W]
      voxel_spacing: tuple with voxel spacing in 3 dimensions
  # Returns:
      Scalar RSD in mm
  """
  
  sd1 = get_surf_dists(y_pred, y_true, voxel_spacing)
  sd2 = get_surf_dists(y_true, y_pred, voxel_spacing)
  
  n_pred = sd1.shape[0]
  n_true = sd2.shape[0]
  
  rsd = np.sqrt( (np.sum(sd1**2) + np.sum(sd2**2)) / (n_true + n_pred) )
  
  return rsd

def msd_max_surf_dist(y_true, y_pred, voxel_spacing=None):
  """
  Calculate Maximum Surface Distance [MSD] for predicted mask. Also known as
  Symmetric Hausdorff Distance
  Expects true and predicted masks to be binary arrays [0/1].
  
  # Arguments:
      y_true: numpy array of true targets, y_true.shape = [slices, H, W]
      y_pred: numpy array of predicted targets, y_pred.shape = [slices, H, W]
      voxel_spacing: tuple with voxel spacing in 3 dimensions
  # Returns:
      Scalar MSD in mm
  """
  
  sd1 = get_surf_dists(y_pred, y_true, voxel_spacing)
  sd2 = get_surf_dists(y_true, y_pred, voxel_spacing)
   
  msd = max(sd1.max(), sd2.max())
  
  return msd

def get_surf_dists(target, reference, voxel_spacing=None):
  """
  Calculate surface distances of target with respect to reference
  
  # Arguments:
      target: numpy array of target mask, target.shape = [slices, H, W]
      reference: numpy array of reference mask, reference.shape = [slices, H, W]
      voxel_spacing: tuple with voxel spacing in 3 dimensions
  # Returns:
      Array with target surface differences in mm
  """
  y_true = reference.astype(np.bool)
  y_pred = target.astype(np.bool)
  
  # 8 connectivity in 3D, any voxel with at least one background voxel as
  # neighbor is part of object boundary
  connectivity_struc_elem = ndi.generate_binary_structure(3, 3)
  
  # Extract boundaries [binary XOR]
  true_border = y_true ^ ndi.binary_erosion(y_true,
                                            structure=connectivity_struc_elem,
                                            iterations=1)
  pred_border = y_pred ^ ndi.binary_erosion(y_pred,
                                            structure=connectivity_struc_elem,
                                            iterations=1)
  # Distance transform, scipy dist. transform returns distance of every
  # non-zero voxel to nearest background, 0, voxel. Hence need to invert
  # border image where the boundary is now 0 and everywhere else is 1 leading
  # to distances over entire image [not signed distance]
  dt = ndi.distance_transform_edt(~true_border, sampling=voxel_spacing)
  surf_dists = dt[pred_border]

  return surf_dists