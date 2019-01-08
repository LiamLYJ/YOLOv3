import os
import numpy as np
import glob
from sklearn.cluster import KMeans
from scipy.misc import imread
import random

def get_new_hw(new_size, h, w):
    if h >=w:
        new_w = new_size * (w / h)
        new_h = new_size
    else:
        new_h = new_size * (h / w)
        new_w = new_size
    return new_h, new_h

def get(new_size = 416, path = '/Dataset/wider_face/train_list_file.txt', save_file_name = 'k_means_data.npy',
        max_blur=1, max_expression=1, max_illumination=1, max_occlusion=2, max_pose=1, max_invalid=1, max_scale = 0.08):
    label_file = os.path.expanduser('~') +  path

    with open(label_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    # need_num = 10000

    data = []
    for img_fn in content:
        # if random.random() > 0.5:
        #     continue
        # if need_num < 1:
        #     break
        # print ('processing.....')
        # need_num -= 1

        img = imread(img_fn)
        h,w,_ = img.shape
        # print (h, w)
        label_fn = img_fn.replace('images', 'labels')
        label_fn = label_fn.replace('.jpg', '.txt')
        # print (label_fn)
        with open(label_fn) as fn:
            label_data = fn.readlines()
            label_data = [x.strip() for x in label_data]
        for each_box in label_data:
            strip_content = [float(x) for x in each_box.split(' ')]

            # not to be count in
            if strip_content[5] > max_blur or  \
                strip_content[6] > max_expression or \
                strip_content[7] > max_illumination or \
                strip_content[8] > max_occlusion or \
                strip_content[9] > max_pose or \
                strip_content[10] > max_invalid or \
                strip_content[4] < max_scale :
                continue

            box_w, box_h = strip_content[3:5]
            new_h, new_w = get_new_hw(new_size, h, w)
            box_w *= new_w
            box_h *= new_h
            data.append([box_w, box_h])
    data = np.array(data)
    print (data.shape)
    np.save(save_file_name, data)


def process(n_clusters = 9, npz_file = 'k_means_data.npy'):
    data = np.load(npz_file)
    kmeans = KMeans(n_clusters = n_clusters, random_state =0).fit(data)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = [[round(x[0]), round(x[1])] for x in cluster_centers]
    cluster_centers = sorted(cluster_centers, key = lambda x : x[0]*x[1])
    print ('center boxes:', cluster_centers)

if __name__ == '__main__':
    use_new = True
    if use_new:
        get(new_size = 224)
    process(n_clusters = 6)
