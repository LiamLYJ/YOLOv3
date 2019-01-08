from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

 
dataset_path = os.path.expanduser('~') + "/Dataset/wider_face/train_list_file.txt"
dataloader = torch.utils.data.DataLoader(
    ListDataset(dataset_path), batch_size=1, shuffle=False, num_workers=1
)

for batch_i, (_, imgs, targets) in enumerate(dataloader):
    targets = np.squeeze(targets.data.numpy(), axis=0)

    # Create plot
    img = imgs[0].data.numpy()
    img = np.transpose(img,(1,2,0))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for class_id, c_x, c_y, box_w, box_h in targets:
        if class_id == 0:
            break
        c_x = (c_x - 0.5 * box_w) * img.shape[0]
        c_y = (c_y - 0.5 * box_h) * img.shape[1]
        box_w = box_w * img.shape[0]
        box_h = box_h * img.shape[1]

        bbox = patches.Rectangle((c_x, c_y), box_w, box_h, linewidth=2,
                        edgecolor='red',
                        facecolor='none')

        ax.add_patch(bbox)

    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig('output/%d.png' % (batch_i), bbox_inches='tight', pad_inches=0.0)
    plt.close()
    if batch_i > 20:
        break

