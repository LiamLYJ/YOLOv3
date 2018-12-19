from __future__ import division

from fix_models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="log", help="path to dataset")

parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs(opt.log_dir, exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)
# model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

writer = SummaryWriter(opt.log_dir)

for epoch in range(opt.epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()
        try:
            loss = model(imgs, targets)
        except:
            print('overflow error, continue to next batch')
            continue

        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                opt.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0)

        if batch_i % 2 == 0:
            iteration = epoch * len(dataloader) + batch_i
            writer.add_scalar('loss_total', loss.item(), iteration)
            writer.add_scalar('loss_x', model.losses["x"], iteration)
            writer.add_scalar('loss_y', model.losses["y"], iteration)
            writer.add_scalar('loss_w', model.losses["w"], iteration)
            writer.add_scalar('loss_h', model.losses["h"], iteration)
            writer.add_scalar('loss_conf', model.losses["conf"], iteration)
            writer.add_scalar('loss_cls', model.losses["cls"], iteration)

            writer.add_scalar('recall', model.losses["recall"], iteration)
            writer.add_scalar('precision', model.losses["precision"], iteration)

            with torch.no_grad():
                detections = model(imgs)
                detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
                # just use the first one from a batch
                for batch_i in range(len(detections)):
                    if detections[batch_i] is not None:
                        img = imgs[batch_i]
                        detection = detections[batch_i]
                        break
                frame = img.data.cpu().numpy()
                frame = 255 * np.transpose(frame, [1,2,0])
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    try:
                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)
                        box_h = y2- y1
                        box_w = x2 - x1
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
                    except:
                        print ('some overflow exception, just skip and continue')
            frame = np.expand_dims(np.transpose(frame, [2,0,1]),0)
            writer.add_image('prediction', frame, iteration)
            original_img = np.expand_dims(img.data.cpu().numpy(), 0)
            writer.add_image('input', original_img, iteration)


    if epoch % opt.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
        model.save_fix_parameters("%s/%d_fix_hyper.pkl" % (opt.checkpoint_dir, epoch))
