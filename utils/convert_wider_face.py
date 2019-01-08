import os
import numpy as np
from PIL import Image

def write_label_file(file_folder, file_path, gt_list):
    os.makedirs(file_folder, exist_ok=True)
    all_path = os.path.join(file_folder, file_path)
    with open(all_path, 'w') as f:
        f.writelines(gt_list)

def convert_widerface(dataset_path, sub_set):
    list_name = "wider_face_{}_bbx_gt.txt".format(sub_set)
    if sub_set == "train":
        sub_folder = "WIDER_train/"
    elif sub_set == "val":
        sub_folder = "WIDER_val/"
    list_path = os.path.join(dataset_path,list_name)
    labels_folder = os.path.join(dataset_path, sub_folder+"labels")
    imgs_folder = os.path.join(dataset_path, sub_folder+"images")
    with open(list_path, 'r') as file:
        all_lines = file.readlines()
        all_lines = [x.rstrip().lstrip() for x in all_lines] # get rid of fringe whitespaces
    gt_list = []
    cur_folder_name = ""
    cur_img_name = ""
    img_width = 0
    img_height = 0
    read_state = 0
    num_gt = 0
    gt_list = []
    img_paths = []
    for item in all_lines:
        if read_state == 0:
            if item.endswith(".jpg"):
                cur_folder_name, cur_img_name = item.split("/")

                image_path = os.path.join(imgs_folder, item)
                img = np.array(Image.open(image_path))
                img_height, img_width, _ = img.shape
                img_paths.append(image_path+"\n")
                read_state = 1
        elif read_state == 1:
            num_gt = int(item)
            read_state = 2
        elif read_state == 2:
            gt_item = item.split(" ")
            gt_item[0] = '{:0.6f}'.format((int(gt_item[0])-1 + 0.5 * float(gt_item[2])) / img_width)
            gt_item[1] = '{:0.6f}'.format((int(gt_item[1])-1 + 0.5 * float(gt_item[3])) / img_height)
            gt_item[2] = '{:0.6f}'.format(int(gt_item[2]) / img_width)
            gt_item[3] = '{:0.6f}'.format(int(gt_item[3]) / img_height)

            gt_list.append("0 " + " ".join(gt_item) + "\n")
            num_gt -= 1
            if num_gt < 1:
                write_label_file(os.path.join(labels_folder,cur_folder_name), cur_img_name.replace(".jpg",".txt"), gt_list)
                gt_list = []
                num_gt = 0
                read_state = 0

    write_label_file(dataset_path, "{}_list_file.txt".format(sub_set), img_paths)

DATASET_PATH = os.path.expanduser('~') + "/Dataset/wider_face/"
convert_widerface(DATASET_PATH, 'train')
convert_widerface(DATASET_PATH, 'val')
