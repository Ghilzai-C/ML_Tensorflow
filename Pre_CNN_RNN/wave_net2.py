import tensorflow as tf 
import matplotlib.pyplot as plt 
from pathlib import Path
import pandas as pd 
import numpy as np


def to_windows(dataset, length):
    dataset = dataset.window(length,  shift=1, drop_remainder=True)
    return dataset.flat_map(lambda window_ds:window_ds.batch(length))


def fit_and_evaluate(model, train_set, valid_set, learning_rate, epochs= 500 ):
    early_stoping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=50, restore_best_weights=True
    )
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9 )
    model.compile(loss = tf.keras.losses.Huber(), optimizer=opt, metrics = ["mae"])
    history = model.fit(train_set, validation_data = valid_set, epochs = epochs, 
                        callbacks = [early_stoping_cb])
    valid_loss, valid_mae = model.evaluate(valid_set)
    return valid_mae *1e6


path = Path("datasets/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv")
df = pd.read_csv(path, parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"]
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1)
df = df.drop_duplicates()  

seq_lenght = 56

df_mulvar = df[["bus", "rail"]] / 1e6 
df_mulvar["next_day_type"] = df["day_type"].shift(-1)
df_mulvar = pd.get_dummies(df_mulvar)

mulvar_train = df_mulvar["2016-01":"2018-12"]
mulvar_train = np.asarray(mulvar_train).astype(np.float32)
mulvar_valid = df_mulvar["2019-01":"2019-05"]
mulvar_valid = np.asarray(mulvar_valid).astype(np.float32)
mulvar_test = df_mulvar["2019-06":]
mulvar_test = np.asarray(mulvar_test).astype(np.float32)

def to_seq2seq_dataset(series, seq_length=56, ahead=14, target_col = 1, 
                       batch_size = 32, shuffle=False, seed = None):
    ds = to_windows(tf.data.Dataset.from_tensor_slices(series), ahead+1)
    ds = to_windows(ds, seq_lenght).map(lambda S: (S[:, 0], S[:,1:, 1]))
    if shuffle:
        ds =ds.shuffle(8 * batch_size , seed=seed)
    return ds.batch(batch_size)

seq2seq_train = to_seq2seq_dataset(mulvar_train, shuffle=True, seed=42)
seq2seq_valid = to_seq2seq_dataset(mulvar_valid)

longer_train = to_seq2seq_dataset(mulvar_train, seq_length=112, shuffle=True, seed = 42)
longer_valid = to_seq2seq_dataset(mulvar_valid, seq_length=112)
downsample_train = longer_train.map(lambda X, Y: (X, Y[:, 3::2])) 
downsample_valid = longer_valid.map(lambda X, Y: (X, Y[:, 3::2]))

tf.random.set_seed(42)

class GatedActivationUnit(tf.keras.layers.Layer):
    def __init__(self, activation = "tanh", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
    def call(self, inputs):
        n_filters = inputs.shape[-1] //2
        linear_output = self.activation(inputs[..., :n_filters])
        gate = tf.keras.activations.sigmoid(inputs[..., n_filters:])
        return self.activation(linear_output) * gate
    
def wavenet_residual_block(inputs, n_filters, dilation_rate):
    z = tf.keras.layers.Conv1D(2 * n_filters, kernel_size=2, padding="causal", 
                                dilation_rate=dilation_rate)(inputs)
    z = GatedActivationUnit()(z)
    z = tf.keras.layers.Conv1D(n_filters, kernel_size=1)(z)
    return tf.keras.layers.Add()([z, inputs]), z
    
tf.random.set_seed(42)
n_layers_per_block = 3
n_blocks = 1
n_filters = 32
n_outputs = 14
inputs = tf.keras.layers.Input(shape=[None,5])
z = tf.keras.layers.Conv1D(n_filters, kernel_size=2, padding="causal")(inputs)
skip_to_last = []
for dilation_rate in [2**i for i in range(n_layers_per_block)]*n_blocks:
    z , skip = wavenet_residual_block(z, n_filters, dilation_rate)
    skip_to_last.append(skip)
z = tf.keras.activations.relu(tf.keras.layers.Add()(skip_to_last))
z = tf.keras.layers.Conv1D(n_filters, kernel_size=1, activation="relu")(z)
y_pred = tf.keras.layers.Conv1D(n_outputs, kernel_size=1)(z)

full_wavenet_model = tf.keras.Model(inputs = [inputs], outputs = [y_pred])
fit_and_evaluate(full_wavenet_model, longer_train, longer_valid, learning_rate=0.1, epochs=5)