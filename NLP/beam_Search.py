
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

def beam_search(sentence_en, beam_width, verbose=False):
    x = np.array([sentence_en])
    x_dec = np.array(["startofseq"])
    y_proba = model.predict((x, x_dec))[0, 0]
    top_k = tf.math.top_k(y_proba, k = beam_width)
    top_translation = [
        (np.log(word_proba), text_vec_layer_es.get_vocabulary()[word_id])
        for word_proba, word_id in zip(top_k.values, top_k.indices)
    ]
    
    if verbose:
        print("top first words : ", top_translation)
    
    for idx in range(1, max_length):
        candidates = []
        for log_proba, translation in top_translation:
            if translation.endswith("endofseq"):
                candidates.append((log_proba, translation))
            x = np.array([sentence_en])
            x_dec = np.array(["startofseq" + translation])
            y_proba = model.predict((x, x_dec))[0, idx]
            
            for word_id, word_proba in enumerate(y_proba):
                word = text_vec_layer_en.get_vocabulary()[word_id]
                candidates.append((log_proba+ np.log(word_proba), f"{translation} {word}"))
        top_translation = sorted(candidates, reverse=True)[:beam_width]
        if verbose:
            print("top translation ", top_translation)
        if all([tr.endswith("endofseq") for _, tr in top_translation]):
            return top_translation[0][1].replace("endofseq", "").strip()
sentence_en = "try to translate it please"
translate(sentence_en)
beam_search(sentence_en, beam_width=3, verbose=True)
