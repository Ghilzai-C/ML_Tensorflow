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


df_mulvar = df[["bus", "rail"]] / 1e6 
df_mulvar["next_day_type"] = df["day_type"].shift(-1)
df_mulvar = pd.get_dummies(df_mulvar)

mulvar_train = df_mulvar["2016-01":"2018-12"]
mulvar_valid = df_mulvar["2019-01":"2019-05"]
mulvar_test = df_mulvar["2019-06":]
seq_length = 56 
tf.random.set_seed(42)

train_mulvar_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_train.to_numpy(),
    targets=mulvar_train[["bus", "rail"]][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_mulvar_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_valid.to_numpy(),
    targets=mulvar_valid[["bus","rail"]][seq_length:],
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

tf.random.set_seed(42)
mulvar_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 5]),
    tf.keras.layers.Dense(1)
])
fit_and_evaluate(mulvar_model, train_mulvar_ds, valid_mulvar_ds, learning_rate=0.05)


bus_naive = mulvar_model.predict(valid_mulvar_ds)
bus_target = mulvar_valid["bus"][seq_length:]
(bus_target - bus_naive).abs().mean() *1e6

y_preds_valid = mulvar_model.predict(valid_mulvar_ds)
for idx, name in enumerate(["bus", "rail"]):
    mae = 1e6 * tf.keras.metrics.mean_absolute_error(
        mulvar_valid[name][seq_length:], y_preds_valid[:, idx]
    )
    print(name, int(mae))
    
    