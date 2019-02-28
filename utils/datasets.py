import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

import sys

def random_distort(
    img,
    brightness_delta=32/255.,
    contrast_delta=0.5,
    saturation_delta=0.5,
    hue_delta=0.1):
    '''A color related data augmentation used in SSD.

    Args:
      img: (PIL.Image) image to be color augmented.
      brightness_delta: (float) shift of brightness, range from [1-delta,1+delta].
      contrast_delta: (float) shift of contrast, range from [1-delta,1+delta].
      saturation_delta: (float) shift of saturation, range from [1-delta,1+delta].
      hue_delta: (float) shift of hue, range from [-delta,delta].

    Returns:
      img: (PIL.Image) color augmented image.
    '''
    def brightness(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(brightness=delta)(img)
        return img

    def contrast(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(contrast=delta)(img)
        return img

    def saturation(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(saturation=delta)(img)
        return img

    def hue(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(hue=delta)(img)
        return img

    img = brightness(img, brightness_delta)
    if random.random() < 0.5:
        img = contrast(img, contrast_delta)
        img = saturation(img, saturation_delta)
        img = hue(img, hue_delta)
    else:
        img = saturation(img, saturation_delta)
        img = hue(img, hue_delta)
        img = contrast(img, contrast_delta)
    return img

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)

class FaceDataset(Dataset):
    def __init__(self, list_path, img_size=416, max_blur=1, max_expression=1, max_illumination=1,
                    max_occlusion=2, max_pose=1, max_invalid=1, max_scale = 0.1, transform = True):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50
        self.max_blur = max_blur
        self.max_expression = max_expression
        self.max_illumination = max_illumination
        self.max_occlusion = max_occlusion
        self.max_pose = max_pose
        self.max_invalid = max_invalid
        self.max_scale = max_scale
        self.transform = transform

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        if self.transform:
            img = random_distort(Image.open(img_path))
        else:
            img = Image.open(img_path)
        img = np.array(img)

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 11)
            # Filter labels by blur, expression, illumination, occlusion, pose and invalid
            # print("before filter: ", labels.shape)
            filter_raw_index = np.where(labels[:, 5] <= self.max_blur)
            labels = labels[filter_raw_index]
            filter_raw_index = np.where(labels[:, 6] <= self.max_expression)
            labels = labels[filter_raw_index]
            filter_raw_index = np.where(labels[:, 7] <= self.max_illumination)
            labels = labels[filter_raw_index]
            filter_raw_index = np.where(labels[:, 8] <= self.max_occlusion)
            labels = labels[filter_raw_index]
            filter_raw_index = np.where(labels[:, 9] <= self.max_pose)
            labels = labels[filter_raw_index]
            filter_raw_index = np.where(labels[:, 10] <= self.max_invalid)
            labels = labels[filter_raw_index]
            filter_raw_index = np.where(labels[:, 3] >= self.max_scale)
            labels = labels[filter_raw_index]
            filter_raw_index = np.where(labels[:, 4] >= self.max_scale)
            labels = labels[filter_raw_index]

            # print("after filter: ", labels.shape)
            labels = labels[:,:5]

            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        # if (np.sum(filled_labels!=0) > 0):
        #     print ('yyyyyyyyyyyyyyyy')
        # else:
        #     print ('nnnnnnnnnnnnnn')

        filled_labels = torch.from_numpy(filled_labels)
        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
