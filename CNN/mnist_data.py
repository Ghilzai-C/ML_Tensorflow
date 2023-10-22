import tensorflow as tf 
import numpy as np


mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, Y_train_full), (X_test, Y_test) = mnist
X_train_full = np.expand_dims(X_train_full.astype(np.float32), axis =-1) / 255
X_test = np.expand_dims(X_test.astype(np.float32), axis=-1) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
Y_train, Y_valid = Y_train_full[:-5000], Y_train_full[-5000:]

# X_train_full_reshape = tf.keras.layers.Resizing(height=224, 
#                                           width = 224, crop_to_aspect_ratio = True)(X_train_full)
# X_test_reshape = tf.keras.layers.Resizing(height=224, 
#                                           width = 224, crop_to_aspect_ratio = True)(X_test)
# X_train_reshape, X_valid_reshape = X_train_full_reshape[:-5000], X_train_full_reshape[-5000:]

