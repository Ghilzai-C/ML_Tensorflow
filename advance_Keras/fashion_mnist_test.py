import tensorflow as tf 


fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, Y_train_full), (X_test, Y_test) = fashion_mnist
X_train, Y_train = X_train_full[:-5000], Y_train_full[:-5000]
X_valid, Y_valid = X_train_full[-5000:], Y_train_full[-5000:]
X_train, X_valid, X_test = X_train/255.0, X_valid/255.0, X_test/255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

tf.keras.backend.clear_session()
tf.random.set_seed(42)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = [28, 28]))
for layer in range(100):
    model.add(tf.keras.layers.Dense(100, activation="relu", 
                                    kernel_initializer= "he_normal"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              metrics=["accuracy"])
history = model.fit(X_train_scaled, Y_train, epochs=2,
                    validation_data=(X_valid_scaled, Y_valid))

# Batch Normalization 
tf.keras.backend.clear_session()
tf.random.set_seed(42)
model_bn = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(300, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation="softmax")
])
model_bn.summary()
print([(var.name, var.trainable) for var in model_bn.layers[1].variables])

model_bn.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",
              metrics=["accuracy"])
model_bn.fit(X_train, Y_train, epochs=2, validation_data=(X_valid, Y_valid))

# Batch Normalization with position change  
tf.keras.backend.clear_session()
tf.random.set_seed(42)
model_bn1 =  tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model_bn1.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", 
              metrics=["accuracy"])
model_bn1.fit(X_train, Y_train, epochs=2, validation_data=(X_valid, Y_valid))