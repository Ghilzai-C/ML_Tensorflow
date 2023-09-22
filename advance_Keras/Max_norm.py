from faster_optimizers import*
from functools import partial
dense = tf.keras.layers.Dense(
    100, activation="relu", kernel_initializer="he_normal",
    kernel_constraint=tf.keras.constraints.max_norm(1.0)
)

MaxNormDense = partial(tf.keras.layers.Dense,
                       activation = "relu", 
                       kernel_constraint = tf.keras.constraints.max_norm(1.0))
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    MaxNormDense(100),
    MaxNormDense(100),
    tf.keras.layers.Dense(10, activation="softmax")
])

optimizer_max = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer=optimizer_max,
              metrics=["accuracy"])
history_max = model.fit(X_train, Y_train, epochs=25, 
                        validation_data=(X_valid, Y_valid))
# time 75.13 acc 88.65
end_time = timeit.default_timer() - start_time
print(end_time)