import tensorflow as tf 
import sklearn
import matplotlib.pyplot as plt 
from sklearn.datasets import load_sample_images
import numpy as np
import matplotlib as mpl


images = load_sample_images()["images"]

# images = tf.keras.layers.CenterCrop(height = 70, width = 120)(images)
images = tf.keras.layers.Rescaling(scale=1 / 255)(images)

mmax_pool = tf.keras.layers.MaxPool2D(pool_size=2)
output = mmax_pool(images)

fig = plt.figure(figsize=(12, 8))
gs = mpl.gridspec.GridSpec(nrows=11,ncols=2,width_ratios=[2, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("input")
ax1.imshow(images[0])
ax1.axis("off")
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("output")
ax2.imshow(output[0])
ax2.axis("off")
# plt.show()

# depth wise pooling

np.random.seed(42)
fmaps = np.random.rand(2, 70, 120, 60)
with tf.device("/cpu:0"):
    output = tf.nn.max_pool(fmaps, ksize=(1,1,1,3), strides=(1,1,1,3), padding="VALID")
print(output.shape)

class DepthPool(tf.keras.layers.Layer):
    def __init__(self, pool_size=2 , **kwargs):
        
        super().__init__(**kwargs)    
        self.pool_size= pool_size
    def call(self, inputs):
        shape = tf.shape(inputs)
        groups = shape[-1] // self.pool_size
        new_shape = tf.concat([shape[:-1],[groups, self.pool_size]], axis=0)
        return tf.reduce_max(tf.reshape(inputs, new_shape),axis=-1)

np.allclose(DepthPool(pool_size=3)(fmaps), output)

depth_output = DepthPool(pool_size=3)(images)

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.title("input")

plt.imshow(images[0])
plt.axis("off")
plt.subplot(1,2,2)
plt.title("output")
plt.imshow(depth_output[0,...,0], cmap="gray")
plt.axis("off")
plt.show()        

