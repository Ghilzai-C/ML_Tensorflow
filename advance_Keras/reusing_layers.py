import tensorflow as tf 
import numpy as np
import timeit 



fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, Y_train_full), (X_test, Y_test) = fashion_mnist
X_train, Y_train = X_train_full[:-5000], Y_train_full[:-5000]
X_valid, Y_valid = X_train_full[-5000:], Y_train_full[-5000:]
X_train, X_valid, X_test = X_train/255.0, X_valid/255.0, X_test/255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

pos_class_id = class_names.index("Pullover")
neg_class_id = class_names.index("T-shirt/top")

# def split_dataset(X, Y):
#     Y_for_b = (Y == pos_class_id) | (Y == neg_class_id)
#     y_a = Y[~Y_for_b]
#     y_b = (Y[Y_for_b] == pos_class_id).astype(np.float32)
#     old_class_ids = list(set(range(10)) - set([neg_class_id, pos_class_id]))
#     for old_class_id , new_class_id in zip(old_class_ids, range(8)):
#         y_a[y_a== old_class_id] = neg_class_id
#     return((X[~Y_for_b], y_a), (X[Y_for_b], y_b))
def split_dataset(X, Y):
    y_for_B = (Y == pos_class_id) | (Y == neg_class_id)
    y_A = Y[~y_for_B]
    y_B = (Y[y_for_B] == pos_class_id).astype(np.float32)
    old_class_ids = list(set(range(10)) - set([neg_class_id, pos_class_id]))
    for old_class_id, new_class_id in zip(old_class_ids, range(8)):
        y_A[y_A == old_class_id] = new_class_id  # reorder class ids for A
    return ((X[~y_for_B], y_A), (X[y_for_B], y_B))


(X_train_A, Y_train_A), (X_train_B, Y_train_B) = split_dataset(X_train, Y_train)
(X_valid_A, Y_valid_A), (X_valid_B, Y_valid_B) = split_dataset(X_valid, Y_valid)
(X_test_A, Y_test_A), (X_test_B, Y_test_B) = split_dataset(X_test, Y_test)
X_train_B = X_train_B[:200]
Y_train_B = Y_train_B[:200]

tf.random.set_seed(42)

model_A = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dense(8, activation="softmax")
])
model_A.compile(loss="sparse_categorical_crossentropy", 
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                metrics=["accuracy"])
history_A = model_A.fit(X_train_A, Y_train_A, epochs=20,
                        validation_data=(X_valid_A, Y_valid_A))

model_A.save("model_A_save")

tf.random.set_seed(42)

model_B = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dense(1, activation="softmax")
])
model_B.compile(loss="binary_crossentropy", 
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                metrics=["accuracy"])
history_A = model_B.fit(X_train_B, Y_train_B, epochs=20,
                        validation_data=(X_valid_B, Y_valid_B))

model_B.save("model_B_save")


model_A = tf.keras.models.load_model("model_A_save")


tf.random.set_seed(42)
model_A_clone = tf.keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())

model_B_on_A = tf.keras.Sequential(model_A.layers[:-1])
model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable=False    
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model_B_on_A.compile(loss="binary_crossentropy", optimizer= optimizer,
                     metrics=["accuracy"])

history_ba = model_B_on_A.fit(X_train_B, Y_train_B, epochs=4,
                              validation_data=(X_valid_B, Y_valid_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model_B_on_A.compile(loss="binary_crossentropy", optimizer= optimizer,
                     metrics=["accuracy"])

history_ba2 = model_B_on_A.fit(X_train_B, Y_train_B, epochs=16,
                              validation_data=(X_valid_B, Y_valid_B))

model_B_on_A.evaluate(X_test_B, Y_test_B)






                



