import tensorflow as tf 
import numpy as np
import timeit 
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
start_time = timeit.default_timer()
et = []
def timing(ct):
    pt = timeit.default_timer() - ct
    if pt >= 60:
        pt /= 60
    print(pt)
    
    return et.append(pt)

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, Y_train_full), (X_test, Y_test) = fashion_mnist
X_train, Y_train = X_train_full[:-5000], Y_train_full[:-5000]
X_valid, Y_valid = X_train_full[-5000:], Y_train_full[-5000:]
X_train, X_valid, X_test = X_train/255.0, X_valid/255.0, X_test/255.0

def build_model(seed=42):
    tf.random.set_seed(seed)
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = [28, 28]),
        tf.keras.layers.Dense(100, activation="relu",
                              kernel_initializer="he_normal"),
        
        tf.keras.layers.Dense(100, activation="relu",
                              kernel_initializer="he_normal"),
        
        tf.keras.layers.Dense(100, activation="relu",
                              kernel_initializer="he_normal"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

def build_and_train_model(optimization):
    model = build_model()
    model.compile(loss="sparse_categorical_crossentropy", optimizer= optimization,
                  metrics=["accuracy"])
    return model.fit(X_train, Y_train, epochs=10, validation_data=(X_valid, Y_valid))

# Momentum optimizer_sgd 
# time taken 29.7s accuracy = 88.54

optimizer_sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

# history_sgd = build_and_train_model(optimizer_sgd)


# Nesterov Accelerated Gradient

optimizer_nag = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
# histroy_nag = build_and_train_model(optimizer_nag)
# time taken 29.12 accuracy = 88.39


# AdaGard 
optimizer_adag = tf.keras.optimizers.Adagrad(learning_rate=0.001)

# history_adag = build_and_train_model(optimizer_adag) 
# time taken 29.33 accuracy = 84.56


# RMSProp
optimizer_RMSp = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
# history_RMSp = build_and_train_model(optimizer_RMSp)
# time taken 34.34 accuracy = 88.36

# Adam
optimizer_ADAM = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
# history_ADAM = build_and_train_model(optimizer_adag)
# time taken 28.8 accuracy = 84.56


# ADaMax 
optimizer_adaMax = tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9,
                                              beta_2=0.999)
# history_adaMax = build_and_train_model(optimizer_adaMax)
# time taken 30.77 accuracy = 90.39

# Nadam
optimizer_Nadam = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9,
                                            beta_2=0.999)
# history_Nadam = build_and_train_model(optimizer_Nadam)
# time taken 46.41 accuracy = 91.07

# AdamW
optimizer_AdamW = tfa.optimizers.AdamW(weight_decay=1e-5, learning_rate=0.001,
                                       beta_1=0.9,beta_2=0.999)
# history_AdamW = build_and_train_model(optimizer_AdamW)
# time taken 31.26  accuracy = 90.69
# for loss in ("loss", "val_loss"):
#     plt.figure(figsize=(12, 8))
#     opt_names = "SGD Momentum Nesterov AdaGard RMSprop Adam Adamax Nadam AdamW"
#     for history, opt_names in zip((history_sgd, history_adag, history_ADAM, history_adaMax, history_AdamW, 
#          history_Nadam, history_RMSp, histroy_nag), opt_names.split()):
#         plt.plot(history.history[loss], label = f"{opt_names}", linewidth=3)
#     plt.grid()
#     plt.xlabel("Epochs")
#     plt.ylabel({"loss":"Training loss", "val_loss":"Validation loss"}[loss])
#     plt.legend(loc="upper left")
#     plt.axis([0, 9 , 0.1, 0.7])
#     plt.show()





# end_time =  timeit.default_timer() -start_time 
# if end_time >= 60:
#     end_time /= 60
# print(end_time)