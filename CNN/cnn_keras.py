import tensorflow as tf 
import sklearn
import matplotlib.pyplot as plt 
from sklearn.datasets import load_sample_images
import numpy as np

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

images = load_sample_images()["images"]
images = tf.keras.layers.CenterCrop(height = 70, width = 120)(images)
images = tf.keras.layers.Rescaling(scale=1 / 255)(images)

print(images.shape)
tf.random.set_seed(42)

# with out padding
# conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7)
# fmaps = conv_layer(images)
# with padding 

conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7,padding="same", strides=2)
fmaps = conv_layer(images)
print(fmaps.shape)

def conv_output_size(input_size, kernel_size, strides=1, padding= "valid"):
    if padding=="valid":
        z = input_size - kernel_size + strides
        output_size = z // strides
        num_ignored = z % strides
        return output_size, num_ignored
    else:
        output_size = (input_size - 1)// strides +1
        num_padded = (output_size -1) * strides + kernel_size - input_size
        return output_size, num_padded
print(conv_output_size(np.array([70, 120]), kernel_size = 7, strides=2, padding= "same"))
# plt.figure(figsize=(15,9))
# for image_idx in (0, 1):
#     for fmap_idx in (0, 1):
#         plt.subplot(2, 2, image_idx * 2 +fmap_idx + 1)
#         plt.imshow(fmaps[image_idx, :, :, fmap_idx], cmap="gray")
#         plt.axis("off")
# plt.show()

kernels, biases = conv_layer.get_weights()
print(kernels.shape)
print(biases.shape)
tf.random.set_seed(42)
filters = tf.random.normal([7,7,3,2])
biases = tf.zeros([2])
fmaps = tf.nn.conv2d(images, filters, strides=1, padding="SAME") + biases

plt.figure(figsize=(15, 9))
filters = np.zeros([7,7,3,2])
filters[:, 3, :, 0] = 1
filters[3, :, :, 1] = 1
fmaps = tf.nn.conv2d(images, filters, strides=1, padding="SAME") + biases

for image_idx in (0, 1):
    for fmap_idx in (0, 1):
        plt.subplot(2, 2, image_idx * 2 + fmap_idx +1)
        plt.imshow(fmaps[image_idx, : , :, fmap_idx], cmap="gray")
        plt.axis("off")
plt.show()

