import tensorflow as tf 
import matplotlib.pyplot as plt 
from pathlib import Path
import pandas as pd 
import numpy as np 

plt.rc('font', size = 14)
plt.rc('axes', labelsize = 14, titlesize= 14)
plt.rc('legend', fontsize = 14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize = 10)

Images_path = Path() / "images" / "nlp"
Images_path.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png",resolution=300):
    path = Images_path / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
# Downloading Shakspare Data 
shakespare_url = "https://homl.info/shakespeare"
file_path = tf.keras.utils.get_file("shakespeare.txt", shakespare_url)
with open(file_path) as f:
    shakespeare_text = f.read()


"".join(sorted(set(shakespeare_text.lower())))
# print(shakespeare_text[:90])

text_vec_layer = tf.keras.layers.TextVectorization(split="character", standardize="lower")

text_vec_layer.adapt([shakespeare_text])

encoded = text_vec_layer([shakespeare_text])[0]

encoded -= 2
n_tokens = text_vec_layer.vocabulary_size() -2
dataset_size = len(encoded)

print(n_tokens)

print(dataset_size)

def to_dataset(sequence, length, shuffle=False, seed = None, batch_size = 32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length +1, shift=1, drop_remainder=True  )
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length +1))
    if shuffle:
        ds = ds.shuffle(100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


print(list (to_dataset(text_vec_layer(["To be"])[0], length=4)))

length = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:1_000_000], length=length, shuffle=True, seed=42)
valid_set = to_dataset(encoded[1_000_000:1_060_000], length=length)

test_set = to_dataset(encoded[1_060_000:], length=length)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(n_tokens, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer = "nadam", metrics=["accuracy"])
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    "my_shakespeare_model", monitor="val_accuracy",save_best_only=True
)
shakespeare_model = tf.keras.Sequential([
    text_vec_layer, 
    tf.keras.layers.Lambda(lambda X: X -2),
    model
])

url = "https://github.com/ageron/data/raw/main/shakespeare_model.tgz"
path = tf.keras.utils.get_file("shakespeare_model.tgz", url, extract=True)
model_path = Path(path).with_name("shakespeare_mode")

def to_dataset_for_stateful_rnn(sequence, length):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length +1, shift= length, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(length+1)).batch(1)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)
stateful_train_set = to_dataset_for_stateful_rnn(encoded[:1_000_000], length)
stateful_valid_set = to_dataset_for_stateful_rnn(encoded[1_000_000:1_060_000], length)
stateful_test_set = to_dataset_for_stateful_rnn(encoded[1_060_000:], length)
list(to_dataset_for_stateful_rnn(tf.range(10), 3))


def to_non_overlapping_windows(sequence, length):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length +1, shift=length, drop_remainder=True)
    return ds.flat_map(lambda window: window.batch(length+1))


def to_batched_dataset_for_stateful_rnn(sequence, length, batch_size = 32):
    parts = np.array_split(sequence, batch_size)
    datasets = tuple(to_non_overlapping_windows(part, length) for part in parts)
    ds = tf.data.Dataset.zip(datasets).map(lambda *windows: tf.stack(windows))
    return ds.map(lambda window: (window[:, :-1], window[:,1:])).prefetch(1)


list(to_batched_dataset_for_stateful_rnn(tf.range(20), length=3, batch_size=2))

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16, batch_input_shape = [1, None]),
    tf.keras.layers.GRU(128, return_sequences=True, stateful=True),
    tf.keras.layers.Dense(n_tokens, activation="softmax")
])

class ResetStatesCallBack(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    "My_stateful_shakespeare_model",
    monitor= "val_accuracy",
    save_best_only=True
)

model.compile(loss = "sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

# history = model.fit(stateful_train_set, validation_data=stateful_valid_set, 
#                     epochs=10, callbacks=[ResetStatesCallBack(), model_ckpt])

stateLess_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(n_tokens, activation="softmax")
])

stateLess_model.build(tf.TensorShape([None, None]))

stateLess_model.set_weights(model.get_weights())

shakespeare_model = tf.keras.Sequential([
    text_vec_layer, 
    tf.keras.layers.Lambda(lambda X: X -2),
    stateLess_model
])
def next_char(text , temperature=1):
    y_proba = shakespeare_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id +2]

def extend_text(text, n_char=50, temperature = 1):
    for _ in range(n_char):
        text += next_char(text, temperature)
    return text 
tf.random.set_seed(42)

print(extend_text("to be or not to be", temperature=0.01))

