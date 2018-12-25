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
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
parser.add_argument("--config_path", type=str, default="config/yolo_lite.cfg", help="path to model config file")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.95, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
parser.add_argument("--test_path", type=str, default='/Dataset/wider_face/val_list_file.txt', help="directory where test path")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints_face/lite", help="directory where model checkpoints are saved"
)
parser.add_argument("--which_one", type=str, default="077", help="which model to load")

opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

# Get data configuration
test_path = os.path.expanduser('~') + opt.test_path

# Initiate model
model = Darknet(opt.config_path)
model, load_epoch = load_model(opt.checkpoint_dir, model, which_one = opt.which_one)
print('scuccese load model, eopch: %d'%(load_epoch))

if cuda:
    model = model.cuda()

model.eval()

# Get dataloader
dataset = ListDataset(test_path, img_size = opt.img_size)
dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print("Compute mAP...")

all_detections = []
all_annotations = []
# 1 for only face detection casses
num_classes = 1

for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
    if batch_i > 100:
        break
    imgs = Variable(imgs.type(Tensor))

    with torch.no_grad():
        outputs = model(imgs)
        outputs = non_max_suppression(outputs, 1, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

    for output, annotations in zip(outputs, targets):

        all_detections.append([np.array([]) for _ in range(num_classes)])
        if output is not None:
            # Get predicted boxes, confidence scores and labels
            pred_boxes = output[:, :5].cpu().numpy()
            scores = output[:, 4].cpu().numpy()
            pred_labels = output[:, -1].cpu().numpy()

            # Order by confidence
            sort_i = np.argsort(scores)
            pred_labels = pred_labels[sort_i]
            pred_boxes = pred_boxes[sort_i]

            for label in range(num_classes):
                all_detections[-1][label] = pred_boxes[pred_labels == label]

        all_annotations.append([np.array([]) for _ in range(num_classes)])
        if any(annotations[:, -1] > 0):

            annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
            _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

            # Reformat to x1, y1, x2, y2 and rescale to image dimensions
            annotation_boxes = np.empty_like(_annotation_boxes)
            annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
            annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
            annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
            annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
            annotation_boxes *= opt.img_size

            for label in range(num_classes):
                all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

average_precisions = {}
for label in range(num_classes):
    # true_positives is not used in simply version evaluation
    true_positives = []
    true_count = 0
    false_count = 0
    scores = []
    num_annotations = 0

    for i in tqdm.tqdm(range(len(all_annotations)), desc=f"Computing AP for class '{label}'"):
        detections = all_detections[i][label]
        annotations = all_annotations[i][label]

        num_annotations += annotations.shape[0]
        detected_annotations = []

        for *bbox, score in detections:
            scores.append(score)

            if annotations.shape[0] == 0:
                true_positives.append(0)
                false_count += 1
                continue

            overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]

            if max_overlap >= opt.iou_thres and assigned_annotation not in detected_annotations:
                true_positives.append(1)
                detected_annotations.append(assigned_annotation)
                true_count += 1
            else:
                true_positives.append(0)
                false_count += 1
    print ('precision: ', 1 - (false_count / (false_count + true_count)))
    print ('recall: ', true_count / num_annotations)

    # # no annotations -> AP for this class is 0
    # if num_annotations == 0:
    #     average_precisions[label] = 0
    #     continue
    #
    # true_positives = np.array(true_positives)
    # false_positives = np.ones_like(true_positives) - true_positives
    # # sort by score
    # indices = np.argsort(-np.array(scores))
    # false_positives = false_positives[indices]
    # true_positives = true_positives[indices]
    #
    # # compute false positives and true positives
    # false_positives = np.cumsum(false_positives)
    # true_positives = np.cumsum(true_positives)
    #
    # # compute recall and precision
    # recall = true_positives / num_annotations
    # precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    #
    # # compute average precision
    # average_precision = compute_ap(recall, precision)
    # average_precisions[label] = average_precision
    #
    # print ('recall: ', recall)
    # print ('precision:', precision)
    # print ('ap: ', average_precision)
