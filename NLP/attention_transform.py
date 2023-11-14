
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

encoder = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(256, return_state=True))

encoder_outputs, *encoder_state = encoder(encoder_embeddings)
encoder_state = [tf.concat(encoder_state[::2], axis=-1), 
                 tf.concat(encoder_state[1::2], axis=-1)]

decoder = tf.keras.layers.LSTM(512, return_sequences=True)
decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)
output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")
y_proba = output_layer(decoder_outputs)

model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs = [y_proba])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
# history = model.fit((x_train, x_train_dec), y_train, epochs=2, validation_data=((x_valid, x_valid_dec), y_valid
                                                                                # ))
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

# positional encoding 
# max_length = 50 
# embed_size = 128 
tf.random.set_seed(42)
pos_embed_layer = tf.keras.layers.Embedding(max_length, embed_size)
batch_max_len_enc = tf.shape(encoder_embeddings)[1]

encoder_in = encoder_embeddings + pos_embed_layer(tf.range(batch_max_len_enc))
batch_max_len_dec = tf.shape(decoder_embeddings)[1]
decoder_in = decoder_embeddings + pos_embed_layer(tf.range(batch_max_len_dec))

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_length, embed_size, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert embed_size % 2 == 0, "embed size must be even"
        p , i = np.meshgrid(np.arange(max_length),
                            2* np.arange(embed_size //2))
        pos_emb = np.empty((1, max_length, embed_size))
        pos_emb[0, :, ::2] = np.sin(p / 10_000 ** (i / embed_size)).T
        pos_emb[0,: , 1::2] = np.cos(p / 10_000 ** (i / embed_size)).T 
        self.pos_encodings = tf.constant(pos_emb.astype(self.dtype))
        self.supports_masking = True
    def call (self, inputs):
        batch_max_length = tf.shape(inputs)[1]
        return inputs + self.pos_encodings[:, :batch_max_length]

pos_embed_layer = PositionalEncoding(max_length, embed_size)
encoder_in = pos_embed_layer(encoder_embeddings)
decoder_in = pos_embed_layer(decoder_embeddings)

figure_max_length = 201
figure_embed_size = 512
pos_emb = ProcessLookupError(figure_max_length, figure_embed_size)
zeros = np.zeros((1, figure_max_length, figure_embed_size), np.float32)
P = pos_emb(zeros)[0].numpy()
i1, i2, crop_i = 100, 101, 150
p1, p2, p3 = 22, 60, 35

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9,5)) 
ax1.plot([p1, p1], [-1, 1], "k--", lable="$p = {}$".format(p1))
ax1.plot([p2, p2], [-1, 1], "k--", lable="$p = {}$".format(p2), alpha=0.5)
ax1.plot(p3, P[p3, i1], "bx", lable="$p = {}$".format(p3))
ax1.plot(P[:, i1], "b-", lable="$p = {}$".format(i1))
ax1.plot(P[:, i2], "b-", lable="$p = {}$".format(i2))
ax1.plot([p1, p2], [P[p1, i1],P[p2, i1] ], "bo")
ax1.plot([p1, p2], [P[p1, i2],P[p2, i2] ], "ro")
ax1.legend(loc="center right", fontsize=14, framealpha=0.95)
ax1.set_ylabel("$P_{(p,i)}$", rotation=0, fontsize=16)
ax1.grid(True, alpha=0.3)
ax1.hlines(0, 0, figure_max_length - 1, color="k", linewidth=1, alpha=0.3)
ax1.axis([0, figure_max_length - 1, -1, 1])
ax2.imshow(P.T[:crop_i], cmap="gray", interpolation="bilinear", aspect="auto")
ax2.hlines(i1, 0, figure_max_length - 1, color="b", linewidth=3)
cheat = 2  # need to raise the red line a bit, or else it hides the blue one
ax2.hlines(i2+cheat, 0, figure_max_length - 1, color="r", linewidth=3)
ax2.plot([p1, p1], [0, crop_i], "k--")
ax2.plot([p2, p2], [0, crop_i], "k--", alpha=0.5)
ax2.plot([p1, p2], [i2+cheat, i2+cheat], "ro")
ax2.plot([p1, p2], [i1, i1], "bo")
ax2.axis([0, figure_max_length - 1, 0, crop_i])
ax2.set_xlabel("$p$", fontsize=16)
ax2.set_ylabel("$i$", rotation=0, fontsize=16)
# save_fig("positional_embedding_plot")
plt.show()





