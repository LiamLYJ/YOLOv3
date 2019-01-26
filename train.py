from __future__ import division

from models import *
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
import math
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="log_face/lite", help="path to dataset")
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolo_lite.cfg", help="path to model config file")
parser.add_argument("--train_path", type=str, default="/Dataset/wider_face/train_list_file.txt", help="path to data config file")
parser.add_argument("--val_path", type=str, default="/Dataset/wider_face/val_list_file.txt", help="path to data config file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints_face/lite", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")

def train(model, epoch, writer):

    model.train()
    train_loss = 0
    train_recall= 0
    train_precision = 0
    train_x_loss = 0
    train_y_loss = 0
    train_w_loss = 0
    train_h_loss = 0
    train_conf_loss = 0
    # Get dataloader
    dataset = FaceDataset(train_path, img_size = opt.img_size, max_scale = 0.03)
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr = model.hyperparams['learning_rate'])

    imgs = None
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

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
        train_loss += loss.item()
        train_recall += model.losses["recall"]
        train_precision += model.losses["precision"]
        train_x_loss += model.losses["x"]
        train_y_loss += model.losses["y"]
        train_w_loss += model.losses["w"]
        train_h_loss += model.losses["h"]
        train_conf_loss += model.losses["conf"]

        model.seen += imgs.size(0)

    iteration = epoch
    num_data = len(dataloader)
    writer.add_scalar('loss_total', train_loss / num_data, iteration)
    writer.add_scalar('recall', train_recall / num_data, iteration)
    writer.add_scalar('precision', train_precision / num_data, iteration)
    writer.add_scalar('loss_x', train_x_loss / num_data, iteration)
    writer.add_scalar('loss_y', train_y_loss / num_data, iteration)
    writer.add_scalar('loss_w', train_w_loss / num_data, iteration)
    writer.add_scalar('loss_h', train_h_loss / num_data, iteration)
    writer.add_scalar('loss_conf', train_conf_loss / num_data, iteration)

    with torch.no_grad():
        # draw detection
        detections = model(imgs)
        detections = non_max_suppression(detections, 1, opt.conf_thres, opt.nms_thres)
        # just use the first one from a batch
        which_one = 0
        for batch_i in range(len(detections)):
            if detections[batch_i] is not None:
                img = imgs[batch_i]
                detection = detections[batch_i]
                which_one = batch_i
                break
        try:
            frame = img.data.cpu().numpy()
        except:
            print('no higher conf than conf thres, just skip the current tfb log')
            return
        frame = 255 * np.transpose(frame, [1,2,0])
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
            try:
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                box_h = y2 - y1
                box_w = x2 - x1
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            except:
                print ('some overflow exception, just skip and continue')

        # draw gt
        frame_gt = img.data.cpu().numpy()
        frame_gt = 255 * np.transpose(frame_gt, [1,2,0])
        frame_gt = np.ascontiguousarray(frame_gt, dtype = np.uint8)
        gts = np.squeeze(targets.cpu().numpy()[which_one,...])
        for _, x1, y1, box_w, box_h in gts:
            x1 = int((x1 - box_w / 2)* opt.img_size)
            y1 = int((y1 - box_h / 2) * opt.img_size)
            box_w = box_w * opt.img_size
            box_h = box_h * opt.img_size
            x2 = min(int(x1 + box_w), opt.img_size)
            y2 = min(int(y1 + box_h), opt.img_size)
            cv2.rectangle(frame_gt, (x1,y1), (x2,y2), (0,255,0), 2)

    frame = np.expand_dims(np.transpose(frame, [2,0,1]),0)
    writer.add_image('prediction', frame, iteration)
    frame_gt = np.expand_dims(np.transpose(frame_gt, [2,0,1]),0)
    writer.add_image('gt', frame_gt, iteration)

def validation(model, epoch, writer):
    model.eval()
    val_loss = 0
    val_recall = 0
    val_precision = 0
    val_x_loss = 0
    val_y_loss = 0
    val_w_loss = 0
    val_h_loss = 0
    val_conf_loss = 0
    # Get dataloader
    dataset = FaceDataset(val_path, img_size = opt.img_size, max_scale = 0.03)
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    imgs = None
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        loss = model(imgs, targets)

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

        val_loss += loss.item()
        val_recall += model.losses["recall"]
        val_precision += model.losses["precision"]
        val_x_loss += model.losses["x"]
        val_y_loss += model.losses["y"]
        val_w_loss += model.losses["w"]
        val_h_loss += model.losses["h"]
        val_conf_loss += model.losses["conf"]

        model.seen += imgs.size(0)

    iteration = epoch
    num_data = len(dataloader)
    test_loss = val_loss / num_data
    writer.add_scalar('val_loss_total', test_loss, iteration)
    writer.add_scalar('val_recall', val_recall / num_data, iteration)
    writer.add_scalar('val_precision', val_precision / num_data, iteration)
    writer.add_scalar('val_loss_x', val_x_loss / num_data, iteration)
    writer.add_scalar('val_loss_y', val_y_loss / num_data, iteration)
    writer.add_scalar('val_loss_w', val_w_loss / num_data, iteration)
    writer.add_scalar('val_loss_h', val_h_loss / num_data, iteration)
    writer.add_scalar('val_loss_conf', val_conf_loss / num_data, iteration)
    global best_loss
    if test_loss < best_loss:
        print("Saving.............")
        best_loss = test_loss
        save_model(opt.checkpoint_dir, epoch, model, best_model=True)

    with torch.no_grad():
        # draw detection
        detections = model(imgs)
        detections = non_max_suppression(detections, 1, opt.conf_thres, opt.nms_thres)
        # just use the first one from a batch
        which_one = 0
        for batch_i in range(len(detections)):
            if detections[batch_i] is not None:
                img = imgs[batch_i]
                detection = detections[batch_i]
                which_one = batch_i
                break
        try:
            frame = img.data.cpu().numpy()
        except:
            print('no higher conf than conf thres, just skip the current tfb log')
            return
        frame = 255 * np.transpose(frame, [1,2,0])
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
            try:
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                box_h = y2 - y1
                box_w = x2 - x1
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            except:
                print ('some overflow exception, just skip and continue')

        # draw gt
        frame_gt = img.data.cpu().numpy()
        frame_gt = 255 * np.transpose(frame_gt, [1,2,0])
        frame_gt = np.ascontiguousarray(frame_gt, dtype = np.uint8)
        gts = np.squeeze(targets.cpu().numpy()[which_one,...])
        for _, x1, y1, box_w, box_h in gts:
            x1 = int((x1 - box_w / 2)* opt.img_size)
            y1 = int((y1 - box_h / 2) * opt.img_size)
            box_w = box_w * opt.img_size
            box_h = box_h * opt.img_size
            x2 = min(int(x1 + box_w), opt.img_size)
            y2 = min(int(y1 + box_h), opt.img_size)
            cv2.rectangle(frame_gt, (x1,y1), (x2,y2), (0,255,0), 2)

        frame = np.expand_dims(np.transpose(frame, [2,0,1]),0)
        writer.add_image('val_prediction', frame, iteration)
        frame_gt = np.expand_dims(np.transpose(frame_gt, [2,0,1]),0)
        writer.add_image('val_gt', frame_gt, iteration)


if __name__ == '__main__':

    opt = parser.parse_args()
    config_path = opt.model_config_path
    print(opt)

    cuda = torch.cuda.is_available() and opt.use_cuda

    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    # Get data configuration
    train_path = os.path.expanduser('~')+ opt.train_path
    val_path = os.path.expanduser('~')+ opt.val_path

   # Initiate model
    module_defs = parse_model_config(config_path)
    hyperparams = module_defs.pop(0)
    model = Darknet(config_path, opt.img_size)
    try:
        model, load_epoch = load_model(opt.checkpoint_dir, model)
        print('scuccese load model, eopch: %d'%(load_epoch))
    except:
        if 'float' in hyperparams['mode']:
            print('initial weight')
            load_epoch = 0
            model.apply(weights_init_normal)
        else:
            try:
                float_dir = opt.checkpoint_dir.rstrip('_fix')
                model, load_epoch = load_model(float_dir, model)
                print('# ########## scuccese load model from float dir:%s, epoch: %d'%(float_dir, load_epoch))
            except:
                raise ValueError('can not initial weigth when training in fix mode')
        
    if cuda:
        model = model.cuda()

    best_loss = float('inf')  # best test loss
    writer = SummaryWriter(opt.log_dir)

    # save_model(opt.checkpoint_dir, 0, model)

    for epoch in range(load_epoch, opt.epochs):
        train(model, epoch, writer)
        validation(model, epoch, writer)
    
        if epoch % opt.checkpoint_interval == 0:
            save_model(opt.checkpoint_dir, epoch, model)
