import tensorflow as tf 
import matplotlib.pyplot as plt 
from pathlib import Path
import pandas as pd 
import numpy as np 
import tensorflow_datasets as tfds

raw_train_set, raw_valid_set, raw_test_set = tfds.load(
    name = "imdb_reviews",
    split=["train[:90%]", "train[90%:]", "test"],
    as_supervised=True
)

tf.random.set_seed(42)

train_set = raw_train_set.shuffle(5000, seed=42).batch(32).prefetch(1)
valid_set = raw_valid_set.batch(32).prefetch(1)
test_set = raw_test_set.batch(32).prefetch(1)


vocab_size = 1000
text_vec_layer = tf.keras.layers.TextVectorization(max_tokens=vocab_size)
text_vec_layer.adapt(train_set.map(lambda reviews, labels: reviews))

embed_size = 128
tf.random.set_seed(42)
model= tf.keras.Sequential([
    text_vec_layer, 
    tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True),
    tf.keras.layers.GRU(128),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

# history = model.fit(train_set, validation_data = valid_set, epochs = 3)

tf.random.set_seed(42)
# manual model
inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)

token_ids = text_vec_layer(inputs)
mask = tf.math.not_equal(token_ids, 0)
z = tf.keras.layers.Embedding(vocab_size, embed_size)(token_ids)
z = tf.keras.layers.GRU(128, dropout=0.2)(z, mask = mask)
outputs= tf.keras.layers.Dense(1, activation="sigmoid")(z)

model_m = tf.keras.Model(inputs = [inputs], outputs= [outputs])

model_m.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
# history = model_m.fit(train_set, validation_data=valid_set, epochs=3)
text_vec_layer_ragged = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size, ragged=True
)

text_vec_layer_ragged.adapt(train_set.map(lambda reviews, labels: reviews))
text_vec_layer_ragged(["Great movie", "This is DiCaprio's best role."])

embed_size =128

tf.random.set_seed(42)
model_r = tf.keras.Sequential([
    text_vec_layer_ragged, 
    tf.keras.layers.Embedding(vocab_size, embed_size),
    tf.keras.layers.GRU(128),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model_r.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model_r.fit(train_set, validation_data=valid_set, epochs=3)







