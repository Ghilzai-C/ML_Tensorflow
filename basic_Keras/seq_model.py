import sys
print(sys.executable)
import tensorflow as tf 
import matplotlib.pyplot as plt
from pathlib import Path
import timeit
import pandas as pd
current_time = timeit.default_timer()
import numpy as np 

Images_path = Path() / "images" / "ann"
# Images_path.mkdir(parents=True, exist_ok=True)

# def save_fig(fig_id, tight_layout = True, fig_extension = "png", resolution = 300):
#     path = Images_path / f"{fig_id}.{fig_extension}"
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format=fig_extension, dpi = resolution)


fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, Y_train_full), (X_test, Y_test) = fashion_mnist
X_train, Y_train = X_train_full[:-5000], Y_train_full[:-5000]
x_valid, y_valid = X_train_full[:-5000], Y_train_full[:-5000]
# print(X_train.shape)
# print(X_train.dtype)
# plt.imshow(X_train[578], cmap="binary")
# plt.axis('off')
# plt.show()
# print(Y_train)
X_train, x_valid, X_test = X_train/255., x_valid/255., X_test/255.
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# print(class_names[Y_train[0]])
m_rows = 8
m_cols = 10
def display_sample_images(n_rows, n_cols):
    plt.figure(figsize=(n_cols*1.2, n_rows*1.2))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(X_train[index],cmap="binary", interpolation="nearest")
            plt.axis('off')
            plt.title(class_names[Y_train[index]])
    plt.subplots_adjust(wspace=0.25, hspace=0.7)
    plt.show()
# display_sample_images(m_rows,m_cols)

# Creating sequential api
def seq_open():
    tf.random.set_seed(42)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[28, 28]))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(300, activation="relu"))
    model.add(tf.keras.layers.Dense(100, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    tf.keras.utils.plot_model(model,"my_fashion_mnist_model.png", show_shapes=True)
    return model.summary() 

# seq_open()
def seq_comp():
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = [28, 28]),
        tf.keras.layers.Dense(300, activation="relu"), 
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.summary()
    hidden1 = model.layers[1]
    weights, biases = hidden1.get_weights()
    # print(
    # tf.keras.utils.plot_model(model, "Fashion_mnist.png", show_shapes=True),
    # model.layers,\
    
    # hidden1.name,\
    
    # weights.shape,\
    # biases,\
    # biases.shape)
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    # model.compile(loss="sparse_categorical_crossentropy",
    #               optimizer="sgd",
    #               metrics=["accuracy"])
    tf.keras.utils.to_categorical([0, 5, 1, 0], num_classes=10)
    np.argmax(
    [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
     [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
    axis=1
    )
    history =model.fit(X_train, Y_train, epochs=30,
                       validation_data=(x_valid, y_valid))
    
    history.params
    print(history.epoch)
    
    pd.DataFrame(history.history).plot(
    figsize=(8,5), xlim=[0, 30], ylim=[0, 1], grid=True, xlabel="Epoch", 
    style=["r--","r--.", "b-", "b-*"]
    )
    plt.legend(loc="lower left")
    plt.show()
    
    # plt.figure(figsize=(8,5))
    # for key, style in zip(histroy.history,["r--", "r--",, "b-", "b-*"] ):
    model.evaluate(X_test, Y_test)
    x_new = X_test[:5]
    y_proba = model.predict(x_new)
    y_proba.round(2)
    y_pred = y_proba.argmax(axis=-1)
    y_pred
    print(np.array(class_names)[y_pred])
    # plt.figure(figsize=(7.5,2.5))
    # for index, image in enumerate(x_new):
    #     plt.subplot(1, 3, index +1)
    #     plt.imshow(image, cmap="binary",interpolation= "nearest")
    #     plt.axis('off')
    #     plt.title(class_names[Y_test[index]])
    # plt.subplots_adjust(wspace=0.2, hspace=0.5)
    # plt.show()
    
        
seq_comp()










final_time = timeit.default_timer() - current_time
if final_time >= 60:
    final_time /= 60
print(final_time)