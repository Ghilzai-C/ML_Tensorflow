import tensorflow as tf 
import matplotlib.pyplot as plt 
from pathlib import Path 
import pandas as pd 


plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend',fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick',labelsize=10)

IMAGES_Path = Path() / "images" / "rnn"
IMAGES_Path.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_Path / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
tf.keras.utils.get_file(
    "ridership.tgz",
    "https://github.com/ageron/data/raw/main/ridership.tgz",
    cache_dir=".",
    extract=True
)

path = Path("datasets/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv")
df = pd.read_csv(path, parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"]
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1)
df = df.drop_duplicates()

# print(df.head())

df["2019-03":"2019-05"].plot(grid=True, marker=".", figsize=(8, 3.5))
save_fig("daily_ridership_plot")
# plt.show()

diff_7 = df[["bus", "rail"]].diff(7)["2019-03":"2019-05"]

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
df.plot(ax=axs[0], legend=False, marker=".")
df.shift(7).plot(ax=axs[0], grid=True, legend=False, linestyle=":")
diff_7.plot(ax=axs[1], grid=True, marker=".")
axs[0].set_ylim([170_000, 900_000])
save_fig("differencing_plot")
# plt.show()

# print(list(df.loc["2019-05-25":"2019-05-27"]["day_type"]))

print(diff_7)

targets = df[["bus", "rail"]]["2019-03":"2019-05"]
print((diff_7 / targets).abs().mean())

# period = slice("2001", "2019")
# df_monthly = df.resample('M').mean()
# rolling_average_12_months = df_monthly[period].rolling(window=12).mean()


# fig. ax = plt.subplots(figsize=(8, 4))
# df_monthly[period].plot(ax=ax, marker=".")
# rolling_average_12_months.plot(ax=ax, grid=True, legend=False)
# save_fig("long_term_ridership_plot")
# plt.show()

from statsmodels.tsa.arima.model import ARIMA

origin, today = "2019-01-01", "2019-05-31"
rail_serie = df.loc[origin:today]["rail"].asfreq("D")
model = ARIMA(rail_serie, order=(1, 0, 0), seasonal_order=(0, 1, 1, 7))
model = model.fit()
y_pred = model.forecast()

y_pred[0]
print(df["rail"].loc["2019-06-01"])

print(df["rail"].loc["2019-05-25"])

origin, start_date, end_date = "2019-01-01", "2019-03-01", "2019-05-31"
time_period = pd.date_range(start_date, end_date)
rail_series = df.loc[origin:end_date]["rail"].asfreq("D")
y_preds = []

# for today in time_period.shift(-1):
#     model = ARIMA(rail_series[origin:today], order=(1, 0, 0), seasonal_order=(0, 1, 1, 7))
#     model = model.fit()
#     y_pred = model.forecast()[0]
#     y_preds.append(y_pred)
    
# y_preds = pd.Series(y_preds, index=time_period)
# mae = (y_preds - rail_series[time_period]).abs().mean()
# print(mae)

# fig, ax = plt.subplots(figsize=(8, 3))
# rail_series.loc[time_period].plot(label="True", ax =ax , marker = ".", grid = True)
# ax.plot(y_preds, color = "r", marker = ".", label = "SARIMA Forcasts")
# plt.legend()
# plt.show()

# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# fig,axs = plt.subplots(nrows=1, ncols= 2, figsize = (15, 5))
# plot_acf(df[period]["rail"], ax = axs[0], lags = 35)
# axs[0].grid()
# plot_pacf(df[period]["rail"], ax = axs[1], lags=35, method="ywm")
# axs[1].grid()
# plt.show()

my_series = [0, 1, 2, 3, 4, 5]
my_dataset = tf.keras.utils.timeseries_dataset_from_array(
    my_series,
    targets=my_series[3:],
    sequence_length= 3,
    batch_size= 2
)
print(list(my_dataset))

for window_dataset in tf.data.Dataset.range(6).window(4, shift =1):
    for element in window_dataset:
        print(f"{element}", end=" ")
    print()

dataset = tf.data.Dataset.range(6).window(4, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window_dataset: window_dataset.batch(4))
for window_tensor in dataset:
    print(f"{window_tensor}")

def to_windows(dataset, length ):
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    return dataset.flat_map(lambda window_ds: window_ds.batch(length))

dataset = to_windows(tf.data.Dataset.range(6), 4)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
print(list (dataset.batch(2)))


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

# tf.random.set_seed(42)
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(1, input_shape=[seq_length])
# ])
# early_stopping_cb = tf.keras.callbacks.EarlyStopping(
#     monitor = "val_mae", patience=50, restore_best_weights=True)
# opt = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)
# model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"])
# history = model.fit(train_ds, validation_data=valid_ds, epochs=500, callbacks=[early_stopping_cb])

# valid_loss, valid_mae = model.evaluate(valid_ds)
# valid_mae *1e6
