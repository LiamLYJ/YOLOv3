import numpy as np
from PIL import Image
from skimage.transform import resize

output_fixpoint_data = {}
image_path = "./data/128.jpg"
img = np.array(Image.open(image_path))
h, w, _ = img.shape
dim_diff = np.abs(h - w)
# Upper (left) and lower (right) padding
pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
# Determine padding
pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
# Add padding
input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
# Resize and normalize
input_img = resize(input_img, (224, 224, 3), mode='reflect')
# Channels-first
input_img = np.transpose(input_img, (2, 0, 1))

output_fixpoint_data["input_image"] = input_img

weights = np.load("./data/weights.npy")
dict_weights = weights.item()
outputs = np.load("./data/outputs.npy")
dict_outputs = outputs.item()
for name, weight in dict_weights.items():
    output_fixpoint_data[name] = weight
for name, output_data in dict_outputs.items():
    output_fixpoint_data[name] = output_data

np.save("./data/fixpoint_data.npy", output_fixpoint_data)
