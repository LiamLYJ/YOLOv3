

PyTorch- YOLOv3 on face detection

fixpoint and float point training and testing

requirements: 
scikit-image
numpy
torch>=0.4.0
torchvision
pillow
matplotlib
tensorboardX

#   BELOW FROM HERE: WHERE TO STEAL

# YOLOv3
Minimal implementation of YOLOv3 in PyTorch for face detection.

## Paper
### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Original Implementation]](https://github.com/pjreddie/darknet)

## Installation
    $ git clone https://github.com/LiamLYJ/YOLOv3.git
    $ cd YOLOv3/
    $ git checkout develop
    $ sudo pip3 install -r require

## Train
```
    train.py [-h] [--epochs EPOCHS] [--image_folder IMAGE_FOLDER]
                [--batch_size BATCH_SIZE]
                [--model_config_path MODEL_CONFIG_PATH]
                [--data_config_path DATA_CONFIG_PATH]
                [--weights_path WEIGHTS_PATH] [--class_path CLASS_PATH]
                [--conf_thres CONF_THRES] [--nms_thres NMS_THRES]
                [--n_cpu N_CPU] [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--checkpoint_dir CHECKPOINT_DIR]
```
## Download Weights
    $ cd weights/
    $ bash download_weights.sh
copy model_800.ckpt to checkpoints_face/lite_fix/

## Test
```
python detect.py --image_folder=data/face_test --config_path=config/yolo_lite_fix.cfg --checkpoint_dir=checkpoints_face/lite_fix --conf_thres=0.98 --which_one=800
```
