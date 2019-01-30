from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolo_lite.cfg', help='path to model config file')
parser.add_argument('--conf_thres', type=float, default=0.99, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.1, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=224, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=False, help='whether to use cuda if available')
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints_face/lite", help="directory where model checkpoints are saved"
)
parser.add_argument("--is_fix_model", type=bool, default=False, help='whether to use fix model')
parser.add_argument("--which_one", type=str, default=None, help="which model to load")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)

# Set up model
if opt.is_fix_model:
    from fix_models import *
    model = Darknet(opt.config_path, img_size=opt.img_size)
    try:
        model, load_epoch = load_model(opt.checkpoint_dir, model, opt.which_one)
        print('# ########## scuccese load model from fix model dir:%s, epoch: %d'%(opt.checkpoint_dir,
                                                                  load_epoch))
    except:
        raise ValueError('can not initial weigth when training in fix mode')
else:
    from models import *
    model = Darknet(opt.config_path, img_size=opt.img_size)
    model, load_epoch = load_model(opt.checkpoint_dir, model, opt.which_one)
    print('scuccese load float model, eopch: %d'%(load_epoch))

if cuda:
    model.cuda()

model.eval() # Set in evaluation mode

dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print ('\nPerforming object detection:')
prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 1, opt.conf_thres, opt.nms_thres)
        print(detections)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)

print ('\nSaving images:')
# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    print ("(%d) Image: '%s'" % (img_i, path))

    # Create plot
    img = np.array(Image.open(path))

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        print(detections)
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
            y2 = int(y1 + box_h)
            x2 = int(x1 + box_w)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imwrite('./output/results_%d.png'%(img_i), img[:,:,::-1])
