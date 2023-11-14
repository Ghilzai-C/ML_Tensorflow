import tensorflow as tf 
import matplotlib.pyplot as plt 
from pathlib import Path
import pandas as pd 
import numpy as np 
import tensorflow_datasets as tfds
import os
import tensorflow_hub as hub

url = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
path = tf.keras.utils.get_file("spa-eng.zip", origin=url, cache_dir="datasets", extract=True)
text = (Path(path).with_name("spa-eng") / "spa.txt").read_text()

text = text.replace("i", "").replace("¿", "")
pairs = [line.split("\t") for line in text.splitlines()]

np.random.seed(42)
np.random.shuffle(pairs)
sentences_en, sentences_es = zip(*pairs)

for i in range(3):
    print(sentences_en[i], "=>", sentences_es[i])

vocab_size = 1000
max_length = 50 
text_vec_layer_en = tf.keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length
)
text_vec_layer_es = tf.keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length
)

text_vec_layer_en.adapt(sentences_en)
text_vec_layer_es.adapt([f"startofseq {s} endofseq" for s in sentences_es])

text_vec_layer_en.get_vocabulary()[:10]

print(text_vec_layer_es.get_vocabulary()[:10])

x_train = tf.constant(sentences_en[:100_000])
x_valid = tf.constant(sentences_en[100_000:])
x_train_dec = tf.constant([f"startofseq {s} endofseq" for s in sentences_en[:100_000]])
x_valid_dec = tf.constant([f"startofseq {s} endofseq" for s in sentences_en[100_000:]])
y_train = text_vec_layer_es([f"startofseq {s} endofseq" for s in sentences_es[:100_000]])
y_valid = text_vec_layer_es([f"startofseq {s} endofseq" for s in sentences_es[100_000:]])

tf.random.set_seed(42)

encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
decoder_inputs = tf.keras.layers.Input(shape=[], dtype = tf.string)

embed_size = 128
encoder_inputs_ids = text_vec_layer_en(encoder_inputs)
decoder_inputs_ids = text_vec_layer_es(decoder_inputs)

encoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
decoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
encoder_embeddings = encoder_embedding_layer(encoder_inputs_ids)
decoder_embeddings = decoder_embedding_layer(decoder_inputs_ids)

encoder = tf.keras.layers.LSTM(512, return_state=True)
encoder_outputs, *encoder_state = encoder(encoder_embeddings)

decoder = tf.keras.layers.LSTM(512, return_sequences=True)
decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)

output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")
y_proba = output_layer(decoder_outputs)

model = tf.keras.Model(inputs = [encoder_inputs, decoder_inputs], 
                       outputs = [y_proba])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
history = model.fit((x_train, x_train_dec), y_train, epochs=2, validation_data=((x_valid, x_valid_dec), y_valid))
def translate(sentence_en):
    translation = ""
    for word_idx in range(max_length):
        x = np.array([sentence_en])
        x_dec = np.array(["startofseq" + translation])
        y_proba = model.predict((x, x_dec))[0, word_idx]
        predicted_word_id = np.argmax(y_proba)
        predicted_word = text_vec_layer_es.get_vocabulary()[predicted_word_id]
        if predicted_word=="endofseq":
            break
        translation += " " + predicted_word
    return translation.strip()
print(translate("test the translation  "))
