from faster_optimizers import*
from functools import partial
layer = tf.keras.layers.Dense(100, activation="relu", 
                              kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))

tf.random.set_seed(42)

RegularizedDense = partial(tf.keras.layers.Dense,
                           activation= "relu",
                           kernel_initializer="he_normal",
                           kernel_regularizer= tf.keras.regularizers.l2(0.01))
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = [28, 28]),
    RegularizedDense(100),
    RegularizedDense(100),
    RegularizedDense(10, activation="softmax")
])

optimizer_regu = tf.keras.optimizers.SGD(learning_rate=0.02)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer_regu, 
              metrics=["accuracy"])
history_relu = model.fit(X_train, Y_train, epochs=25, validation_data=(X_valid, Y_valid))
# time 69.59 acc 82.063
end_time = timeit.default_timer() - start_time
print(end_time)