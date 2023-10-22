import tensorflow as tf 
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np


dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes

test_set_raw, valid_set_raw, train_set_raw = tfds.load("tf_flowers",
                                                       split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
                                                       as_supervised=True)
tf.keras.backend.clear_session()

batch_size = 32
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=224,width=224, crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])
train_set = train_set_raw.map(lambda X, y:(preprocess(X), y))
train_set = train_set.shuffle(1000, seed = 42).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)
test_set = test_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)

tf.random.set_seed(42)
base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
loc_output = tf.keras.layers.Dense(4)(avg)
class_output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs= base_model.input, outputs = [class_output, loc_output])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(loss=["sparse_categorical_crossentropy", "mse"], loss_weights=[0.8, 0.2],
              optimizer=optimizer, metrics=["accuracy"])

def add_random_bounding_boxes(images, labels):
    fake_boxes = tf.random.uniform([tf.shape(images)[0], 4])
    return images, (labels, fake_boxes)

fake_train_set = train_set.take(5).repeat(2).map(add_random_bounding_boxes)
model.fit(fake_train_set, epochs=10)
