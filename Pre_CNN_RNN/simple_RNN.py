import tensorflow as tf 
import matplotlib.pyplot as plt 
from pathlib import Path
import pandas as pd 



path = Path("datasets/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv")
df = pd.read_csv(path, parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"]
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1)
df = df.drop_duplicates()

rail_train = df["rail"]["2016-01":"2018-12"] / 1e6
rail_valid = df["rail"]["2019-01":"2019-05"] / 1e6
rail_test = df["rail"]["2019-06":] / 1e6
seq_length = 56 
tf.random.set_seed(42)
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    rail_train.to_numpy(),
    targets= rail_train[seq_length:],
    sequence_length= seq_length,
    batch_size=32, 
    shuffle=True,
    seed=42
)
valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    rail_valid.to_numpy(),
    targets=rail_valid[seq_length:],
    sequence_length=seq_length,
    batch_size=32
)

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
# fit_and_evaluate(model, train_ds, valid_ds, learning_rate=0.02)

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape = [None, 1])
])

tf.random.set_seed(42)
univar_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 1]),
    tf.keras.layers.Dense(1)
])
# fit_and_evaluate(univar_model, train_ds, valid_ds, learning_rate=0.05)

# Deep RNNs 
tf.random.set_seed(42)
deep_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1)
])
fit_and_evaluate(deep_model, train_ds, valid_ds, learning_rate=0.01)

