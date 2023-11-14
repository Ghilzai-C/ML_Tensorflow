
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

encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
decoder_inputs = tf.keras.layers.Input(shape=[], dtype = tf.string)

embed_size = 128
encoder_inputs_ids = text_vec_layer_en(encoder_inputs)
decoder_inputs_ids = text_vec_layer_es(decoder_inputs)

encoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
decoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
encoder_embeddings = encoder_embedding_layer(encoder_inputs_ids)
decoder_embeddings = decoder_embedding_layer(decoder_inputs_ids)

tf.random.set_seed(42)
pos_embed_layer = tf.keras.layers.Embedding(max_length, embed_size)
batch_max_len_enc = tf.shape(encoder_embeddings)[1]

encoder_in = encoder_embeddings + pos_embed_layer(tf.range(batch_max_len_enc))
batch_max_len_dec = tf.shape(decoder_embeddings)[1]
decoder_in = decoder_embeddings + pos_embed_layer(tf.range(batch_max_len_dec))

N = 2
num_heads = 8 
dropout_rate = 0.1
n_units = 128 
encoder_pad_mask = tf.math.not_equal(encoder_inputs_ids, 0)[:, tf.newaxis]
z = encoder_in
for _ in range(N):
    skip = z 
    attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_size, dropout= dropout_rate
    )
    z = attn_layer(z, value = z, attention_mask = encoder_pad_mask)
    z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([z, skip]))
    skip = z
    z = tf.keras.layers.Dense(n_units, activation="relu")(z)
    z = tf.keras.layers.Dense(embed_size)(z)
    z = tf.keras.layers.Dropout(dropout_rate)(z)
    z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([z, skip]))

decoder_pad_mask = tf.math.not_equal(decoder_inputs_ids, 0)[:, tf.newaxis]
casual_mask = tf.linalg.band_part(tf.ones((batch_max_len_dec,batch_max_len_dec), tf.bool), -1, 0)

encoder_outputs = z
z = decoder_in
for _ in range(N):
    skip = z
    atten_layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_size,
                                                     dropout=dropout_rate)
    z = atten_layer(z, value = z, attention_mask = casual_mask & decoder_pad_mask)
    z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([z, skip]))
    skip = z
    atten_layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_size,
                                                     dropout=dropout_rate)
    z = atten_layer(z , value= encoder_outputs, attention_mask = encoder_pad_mask)
    z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([z, skip]))
    skip = z
    z = tf.keras.layers.Dense(n_units, activation="relu")(z)
    z = tf.keras.layers.Dense(embed_size)(z)
    z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([z, skip]))
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


y_proba = tf.keras.layers.Dense(vocab_size, activation="softmax")(z)
model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs = [y_proba])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

history = model.fit((x_train, x_train_dec), y_train, epochs=10, validation_data=((x_valid, x_valid_dec), y_valid))
model.save()
print(translate("try to translate it in spanish good luck"))