from pathlib import Path
import tensorflow as tf 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import shutil
from time import strftime

def get_run_logDir(root_logdir="my_log"):
    return Path(root_logdir)/ strftime("run_%y_%m_%d_%H_%M_%S")
run_logDir = get_run_logDir()

import tensorboard
def earthquake_data():
    data_path = Path(r"C:\Users\kashi\Desktop\ML\Earthquake\Eartquakes-1990-2023.csv")
    
    if not data_path.is_file:
        print("can not open data please check the file path ")
    return pd.read_csv(data_path)

earthquake = earthquake_data()
# print(earthquake.info())
# print(earthquake.describe())
earthquake_sd = earthquake.drop("magnitudo",axis=1)
earthquake_sd = earthquake_sd.drop("place",axis=1)
earthquake_sd = earthquake_sd.drop("data_type",axis=1)
earthquake_sd = earthquake_sd.drop("state",axis=1)
earthquake_sd = earthquake_sd.drop("date",axis=1)
earthquake_sd = earthquake_sd.drop("status",axis=1)
earthquake_labels = earthquake["magnitudo"].copy()
print(earthquake_sd.shape)
X_train_full, X_test, y_train_full, y_test = train_test_split(earthquake_sd,  earthquake_labels, random_state = 42)

X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full, y_train_full, random_state = 42)

tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(50, activation = "relu"),
    tf.keras.layers.Dense(50, activation = "relu"),
    tf.keras.layers.Dense(50, activation = "relu"),

    tf.keras.layers.Dense(1)
])
model.summary()



optmizer_o = tf.keras.optimizers.Adamax(learning_rate=1e-3)
model.compile(loss="mse",optimizer=optmizer_o, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)

tensorBoard_cb = tf.keras.callbacks.TensorBoard(run_logDir, profile_batch=(100, 200))



history = model.fit(X_train, Y_train, epochs=5,
                  validation_data=(X_valid, Y_valid), callbacks=[tensorBoard_cb])
mse_test, rmse_test= model.evaluate(X_test, y_test)
print("my_logs")
for path in sorted(Path("my_logs").glob("**/*")):
    print(" " * (len(path.parts) - 1)+path.parts[-1])
    
    


# optimizar = tf.keras.optimizers.Adam(learning_rate=1e-2)
# model.compile(loss="mse", optimizer = optimizar, metrics=["RootMeanSquaredError"])
# norm_layer.adapt(X_train)

x_new = X_test[:3]
y_pred = model.predict(x_new)
print(rmse_test)
print(y_pred)
shutil.rmtree("model_saved_via_shutil", ignore_errors= True)
# saving as tf
model.save("model_SGD_via_tf", save_format="tf")

for path in sorted(Path().glob("model_SGD_via_tf.*")):
    print(path)
for path2 in sorted(Path().glob("model_saved_via_shutil.*")):
    print(path2)

# print(len(train_set))
# print(train_set[:].isnull().sum())
# print(min(earthquake["magnitudo"]))
# earthquake["magnitud_bin"] = pd.cut(earthquake["magnitudo"],
#                                     bins = [-9.99, -6., -2., 0 , 2., 6., 9.1, np.inf ],
#                                     labels = [1, 2, 3, 4, 5, 6, 7])
# earthquake["magnitud_bin"].value_counts().sort_index().plot.bar(rot=0, grid=True)
# plt.xlabel("magnitude")
# plt.ylabel("numbers of intensity")
# plt.show()
# e1arthquake = train_set.copy()
# e1arthquake.plot(kind = "scatter", x = "longitude", y = "latitude", grid = True)
# plt.show()

# e1arthquake.plot(kind = "scatter", x = "longitude", y = "latitude", grid = True, alpha = 0.075)
# plt.show()
# num_data = e1arthquake[["status"]]
# print(num_data.head(20))
# corr_matrix = e1arthquake.corr()
# corr_matrix["significance"].sort_value(ascending = False)







# sig_data = [earthquake["time"], earthquake["significance"],
#             earthquake["tsunami"], earthquake["longitude"], earthquake["latitude"],
#             earthquake[ "depth"]]
# print(sig_data)
# print(earthquake.shape)
# print(sig_data.shape())
# X_train_full, X_test, Y_train_full, Y_test = train_test_split(
#     sig_data, 
#     earthquake["magnitudo"], random_state = 42)
# X_train, X_valid, Y_train, Y_valid = train_test_split(
#     X_train_full, Y_train_full, random_state=42
# )

