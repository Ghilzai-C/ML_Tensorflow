import tensorflow as tf 
import matplotlib.pyplot as plt 
from pathlib import Path
import pandas as pd 
import numpy as np

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

def to_windows(dataset, length):
    dataset = dataset.window(length,  shift=1, drop_remainder=True)
    return dataset.flat_map(lambda window_ds:window_ds.batch(length))

IMAGES_Path = Path() / "images" / "rnn"
IMAGES_Path.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_Path / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

tf.random.set_seed(42)
univar_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 1]),
    tf.keras.layers.Dense(1)
])

path = Path("datasets/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv")
df = pd.read_csv(path, parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"]
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1)
df = df.drop_duplicates()       
# df = np.asarray(df).astype(np.float32)

rail_train = df["rail"]["2016-01":"2018-12"] / 1e6
rail_valid = df["rail"]["2019-01":"2019-05"] / 1e6
rail_test = df["rail"]["2019-06":] / 1e6
seq_lenght = 56

df_mulvar = df[["bus", "rail"]] / 1e6 
df_mulvar["next_day_type"] = df["day_type"].shift(-1)
df_mulvar = pd.get_dummies(df_mulvar)

mulvar_train = df_mulvar["2016-01":"2018-12"]
mulvar_train = np.asarray(mulvar_train).astype(np.float32)
mulvar_valid = df_mulvar["2019-01":"2019-05"]
mulvar_valid = np.asarray(mulvar_valid).astype(np.float32)
mulvar_test = df_mulvar["2019-06":]


X = rail_valid.to_numpy()[np.newaxis, :seq_lenght, np.newaxis]
for step_ahead in range(14):
    y_pred_one = univar_model.predict(X)
    X = np.concatenate([X, y_pred_one.reshape(1,1,1)], axis=1)

Y_pred= pd.Series(X[0, -14:, 0],
                  index=pd.date_range("2019-02-26", "2019-03-11"))   
fig, ax = plt.subplots(figsize= (8, 3.5))
(rail_valid * 1e6)["2019-02-01":"2019-03-11"].plot(
    label="True", marker = ".", ax = ax
)
(Y_pred * 1e6).plot(label = "Predictions", grid=True, marker = "x", color = "r", ax=ax)
ax.vlines("2019-02-25", 0, 1e6, color = "k", linestyle="--", label = "Today")

ax.set_ylim([200_000, 900_000])
plt.legend(loc="center left")
save_fig("forecast_ahead_plot")
# plt.show()


tf.random.set_seed(42)

def split_inputs_and_targets(mulvar_series, ahead=14, target_col = 1):
    return mulvar_series[:, :-ahead], mulvar_series[:, -ahead:, target_col]

# ahead_train_ds = tf.keras.utils.timeseries_dataset_from_array(
#     mulvar_train.to_numpy(),
#     targets=None,
#     sequence_length=seq_lenght +14,
#     batch_size=32, 
#     shuffle=True,
#     seed=42
# ).map(split_inputs_and_targets)

# ahead_valid_ds = tf.keras.utils.timeseries_dataset_from_array(
#     mulvar_valid.to_numpy(),
#     targets=None,
#     sequence_length=seq_lenght+ 14,
#     batch_size=32
# ).map(split_inputs_and_targets)

tf.random.set_seed(42)

ahead_Model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape= [None, 5]),
    tf.keras.layers.Dense(14)
])

# fit_and_evaluate(ahead_Model, ahead_train_ds, ahead_valid_ds, learning_rate=0.02)

# X = mulvar_valid.to_numpy()[np.newaxis, : seq_lenght]
# Y_pred = ahead_Model.predict(X)

my_series = tf.data.Dataset.range(7)
dataset = to_windows(to_windows(my_series, 3), 4)

dataset = dataset.map(lambda S: (S[:, 0], S[:, 1]))

print(list(dataset))

def to_seq2seq_dataset(series, seq_length=56, ahead=14, target_col = 1, 
                       batch_size = 32, shuffle=False, seed = None):
    ds = to_windows(tf.data.Dataset.from_tensor_slices(series), ahead+1)
    ds = to_windows(ds, seq_lenght).map(lambda S: (S[:, 0], S[:,1:, 1]))
    if shuffle:
        ds =ds.shuffle(8 * batch_size , seed=seed)
    return ds.batch(batch_size)

seq2seq_train = to_seq2seq_dataset(mulvar_train, shuffle=True, seed=42)
seq2seq_valid = to_seq2seq_dataset(mulvar_valid)

tf.random.set_seed(42)


seq2seq_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32,return_sequences=True, input_shape=[None, 5]),
    tf.keras.layers.Dense(14)
])
fit_and_evaluate(seq2seq_model, seq2seq_train, seq2seq_valid, learning_rate=0.1)

X = mulvar_valid.to_numpy()[np.newaxis, :seq_lenght]
y_pred_14 = seq2seq_model.predict(x)[0, -1]
Y_pred_valid = seq2seq_model.predict(seq2seq_valid)
for ahead in range(14):
    preds = pd.Series(Y_pred_valid[:-1, -1, ahead], 
                      index= mulvar_valid.index[56 + ahead : -14 +ahead])
    mae = (preds - mulvar_valid["rail"]).abs().mean()*1e6
    print(f"mea for + {ahead +1} : {mae:,.0f}")
    