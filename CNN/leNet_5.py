from mnist_data import*
from functools import partial

tf.random.set_seed(42)
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size= 5, padding="same",
                        activation="tanh", strides=1, kernel_initializer="he_normal")
model = tf.keras.Sequential([
    DefaultConv2D(filters=64, input_shape = [28,28,1]),
    tf.keras.layers.AveragePooling2D(kernel_size=2, stride=2,),
    DefaultConv2D(filters=128),
    tf.keras.layers.AveragePooling2D(kernel_size=2, stride=2,),
    DefaultConv2D(filters=256),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=84, activation="relu",
                          kernel_initializer="he_normal"),
    
    tf.keras.layers.Dense(units=10, activation="rbf")
    
])

model.compile(loss = "sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model.summary()

history = model.fit(X_train, Y_train, epochs=10,
                    validation_data=(X_valid, Y_valid))
score = model.evaluate(X_test, Y_test)
x_new = X_test[:10]
y_pred = model.predict(x_new)