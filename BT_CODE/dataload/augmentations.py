import numpy as np
import torch
import copy
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
import SimpleITK as sitk
import pickle
from scipy.ndimage import gaussian_filter
import random
import abc
import os


def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


class AbstractTransform(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str


def uniform(low, high, size=None):

    if low == high:
        if size is None:
            return low
        else:
            return np.ones(size) * low
    else:
        return np.random.uniform(low, high, size)


def augment_contrast(data_sample, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
    if not per_channel:
        mn = data_sample.mean()
        if preserve_range:
            minm = data_sample.min()
            maxm = data_sample.max()
        if np.random.random() < 0.5 and contrast_range[0] < 1:
            factor = np.random.uniform(contrast_range[0], 1)
        else:
            factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
        data_sample = (data_sample - mn) * factor + mn
        if preserve_range:
            data_sample[data_sample < minm] = minm
            data_sample[data_sample > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            mn = data_sample[c].mean()
            if preserve_range:
                minm = data_sample[c].min()
                maxm = data_sample[c].max()
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
            data_sample[c] = (data_sample[c] - mn) * factor + mn
            if preserve_range:
                data_sample[c][data_sample[c] < minm] = minm
                data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp
    shp = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_seg'] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict):
    shp = data_dict['orig_shape_data']
    current_shape = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    shp = data_dict['orig_shape_seg']
    current_shape_seg = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    return data_dict


def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError("value must be either a single vlaue or a list/tuple of len 2")
        return n_val
    else:
        return value


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


def augment_gaussian_blur(data_sample, sigma_range, per_channel=True, p_per_channel=1):
    if not per_channel:
        sigma = get_range_val(sigma_range)
    for c in range(data_sample.shape[0]):
        if np.random.uniform() <= p_per_channel:
            if per_channel:
                sigma = get_range_val(sigma_range)
            data_sample[c] = gaussian_filter(data_sample[c], sigma, order=0)
    return data_sample


def augment_brightness_multiplicative(data_sample, multiplier_range=(0.5, 2), per_channel=True):
    multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
    if not per_channel:
        data_sample *= multiplier
    else:
        for c in range(data_sample.shape[0]):
            multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
            data_sample[c] *= multiplier
    return data_sample


def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats=False):
    if invert_image:
        data_sample = - data_sample
    if not per_channel:
        if retain_stats:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats:
            data_sample = data_sample - data_sample.mean() + mn
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
    else:
        for c in range(data_sample.shape[0]):
            if retain_stats:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats:
                data_sample[c] = data_sample[c] - data_sample[c].mean() + mn
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
    if invert_image:
        data_sample = - data_sample
    return data_sample


def augment_linear_downsampling_scipy(data_sample, zoom_range=(0.5, 1), per_channel=True, p_per_channel=1,
                                      channels=None, order_downsample=1, order_upsample=0, ignore_axes=None):
    if not isinstance(zoom_range, (list, tuple, np.ndarray)):
        zoom_range = [zoom_range]

    shp = np.array(data_sample.shape[1:])
    dim = len(shp)

    if not per_channel:
        if isinstance(zoom_range[0], (tuple, list, np.ndarray)):
            assert len(zoom_range) == dim
            zoom = np.array([uniform(i[0], i[1]) for i in zoom_range])
        else:
            zoom = uniform(zoom_range[0], zoom_range[1])

        target_shape = np.round(shp * zoom).astype(int)

        if ignore_axes is not None:
            for i in ignore_axes:
                target_shape[i] = shp[i]

    if channels is None:
        channels = list(range(data_sample.shape[0]))

    for c in channels:
        if np.random.uniform() < p_per_channel:
            if per_channel:
                if isinstance(zoom_range[0], (tuple, list, np.ndarray)):
                    assert len(zoom_range) == dim
                    zoom = np.array([uniform(i[0], i[1]) for i in zoom_range])
                else:
                    zoom = uniform(zoom_range[0], zoom_range[1])

                target_shape = np.round(shp * zoom).astype(int)
                if ignore_axes is not None:
                    for i in ignore_axes:
                        target_shape[i] = shp[i]

            downsampled = resize(data_sample[c].astype(float), target_shape, order=order_downsample, mode='edge',
                                 anti_aliasing=False)
            data_sample[c] = resize(downsampled, shp, order=order_upsample, mode='edge',
                                    anti_aliasing=False)

    return data_sample


def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2)):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]")
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data, sample_seg


class Compose(AbstractTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"


class GaussianNoiseTransform(AbstractTransform):
    def __init__(self, noise_variance=(0, 0.1), data_key="data", label_key="seg", p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.noise_variance = noise_variance
    
    def  __call__(self, img_array):
        if np.random.uniform() < self.p_per_sample:
            img_array = augment_gaussian_noise(img_array, self.noise_variance)
        return img_array

    # def __call__(self, data_dict):
    #     for b in range(len(data_dict[self.data_key])):
    #         if np.random.uniform() < self.p_per_sample:
    #             data_dict[self.data_key][b] = augment_gaussian_noise(data_dict[self.data_key][b], self.noise_variance)
    #     return data_dict


class GaussianBlurTransform(AbstractTransform):
    def __init__(self, blur_sigma=(1, 5), data_key="data", label_key="seg", different_sigma_per_channel=True,
                 p_per_channel=1, p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.label_key = label_key
        self.blur_sigma = blur_sigma
    
    def  __call__(self, img_array):
        if np.random.uniform() < self.p_per_sample:
            img_array = augment_gaussian_blur(img_array,  self.blur_sigma,
                        self.different_sigma_per_channel, self.p_per_channel)
        return img_array

    # def __call__(self, data_dict):
    #     for b in range(len(data_dict[self.data_key])):
    #         if np.random.uniform() < self.p_per_sample:
    #             data_dict[self.data_key][b] = augment_gaussian_blur(data_dict[self.data_key][b], self.blur_sigma,
    #                                                              self.different_sigma_per_channel, self.p_per_channel)
    #     return data_dict


class BrightnessMultiplicativeTransform(AbstractTransform):
    def __init__(self, multiplier_range=(0.5, 2), per_channel=True, data_key="data", p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel
    
    def  __call__(self, img_array):
        if np.random.uniform() < self.p_per_sample:
            img_array = augment_brightness_multiplicative(img_array,self.multiplier_range, self.per_channel)
        return img_array

    # def __call__(self, data_dict):
    #     for b in range(len(data_dict[self.data_key])):
    #         if np.random.uniform() < self.p_per_sample:
    #             data_dict[self.data_key][b] = augment_brightness_multiplicative(data_dict[self.data_key][b],
    #                                                                             self.multiplier_range,
    #                                                                             self.per_channel)
    #     return data_dict


class GammaTransform(AbstractTransform):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="data", retain_stats=False,
                 p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image
    
    def  __call__(self, img_array):
        if np.random.uniform() < self.p_per_sample:
            img_array = augment_gamma(img_array,self.gamma_range,
                                                            self.invert_image,
                                                            per_channel=self.per_channel,
                                                            retain_stats=self.retain_stats)
        return img_array

    # def __call__(self, data_dict):
    #     for b in range(len(data_dict[self.data_key])):
    #         if np.random.uniform() < self.p_per_sample:
    #             data_dict[self.data_key][b] = augment_gamma(data_dict[self.data_key][b], self.gamma_range,
    #                                                         self.invert_image,
    #                                                         per_channel=self.per_channel,
    #                                                         retain_stats=self.retain_stats)
    #     return data_dict


class SimulateLowResolutionTransform(AbstractTransform):
    def __init__(self, zoom_range=(0.5, 1), per_channel=False, p_per_channel=1,
                 channels=None, order_downsample=1, order_upsample=0, data_key="data", p_per_sample=1,
                 ignore_axes=None):
        self.order_upsample = order_upsample
        self.order_downsample = order_downsample
        self.channels = channels
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.zoom_range = zoom_range
        self.ignore_axes = ignore_axes
    
    def  __call__(self, img_array):
        if np.random.uniform() < self.p_per_sample:
            img_array = augment_linear_downsampling_scipy(img_array,
                                                            zoom_range=self.zoom_range,
                                                            per_channel=self.per_channel,
                                                            p_per_channel=self.p_per_channel,
                                                            channels=self.channels,
                                                            order_downsample=self.order_downsample,
                                                            order_upsample=self.order_upsample,
                                                            ignore_axes=self.ignore_axes)
        return img_array

    # def __call__(self, data_dict):
    #     for b in range(len(data_dict[self.data_key])):
    #         if np.random.uniform() < self.p_per_sample:
    #             data_dict[self.data_key][b] = augment_linear_downsampling_scipy(data_dict[self.data_key][b],
    #                                                                             zoom_range=self.zoom_range,
    #                                                                             per_channel=self.per_channel,
    #                                                                             p_per_channel=self.p_per_channel,
    #                                                                             channels=self.channels,
    #                                                                             order_downsample=self.order_downsample,
    #                                                                             order_upsample=self.order_upsample,
    #                                                                             ignore_axes=self.ignore_axes)
    #     return data_dict


class ContrastAugmentationTransform(AbstractTransform):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, data_key="data",
                 p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
    
    def  __call__(self, img_array):
        if np.random.uniform() < self.p_per_sample:
            img_array = augment_contrast(img_array,contrast_range=self.contrast_range,
                                                               preserve_range=self.preserve_range,
                                                               per_channel=self.per_channel)
        return img_array

    # def __call__(self, data_dict):
    #     for b in range(len(data_dict[self.data_key])):
    #         if np.random.uniform() < self.p_per_sample:
    #             data_dict[self.data_key][b] = augment_contrast(data_dict[self.data_key][b],
    #                                                            contrast_range=self.contrast_range,
    #                                                            preserve_range=self.preserve_range,
    #                                                            per_channel=self.per_channel)
    #     return data_dict


class MirrorTransform(AbstractTransform):
    def __init__(self, axes=(0, 1, 2), data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        for b in range(len(data)):
            sample_seg = None
            if seg is not None:
                sample_seg = seg[b]
            ret_val = augment_mirroring(data[b], sample_seg, axes=self.axes)
            data[b] = ret_val[0]
            if seg is not None:
                seg[b] = ret_val[1]

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict
