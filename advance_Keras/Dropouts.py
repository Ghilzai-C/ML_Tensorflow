from faster_optimizers import*
import numpy as np 
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = [28, 28]),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

optimizer_dropout = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer_dropout,
              metrics=["accuracy"])
# history_dropout = model.fit(X_train, Y_train, epochs=25, 
#                             validation_data=(X_valid, Y_valid))
# time 75.58 acc 89.063
end_time = timeit.default_timer() - start_time
print(end_time)
model.evaluate(X_train, Y_train)
model.evaluate(X_test, Y_test)


# MC dropouts 
tf.random.set_seed(42)

y_probas = np.stack([model(X_test, training= True) for sample in range(100)])
y_proba = y_probas.mean(axis=0)

model.predict(X_test[:1]).round(5)

y_proba[0].round(5)
y_std = y_probas.std(axis=0)
y_std[0].round(5)

y_pred = y_proba.argmax(axis=1)
accuracy = (y_pred == Y_test).sum() /len(Y_test)

class MCDropouts(tf.keras.layers.Dropout):
    def call(self, inputs, training= None):
        return super(). call(inputs, training=True)

Dropouts = tf.keras.layers.Dropout
mc_drop = tf.keras.Sequential([
    MCDropouts(layer.rate) if isinstance(layer, Dropouts) else layer 
    for layer in model.layers
])
mc_drop.set_weights(model.get_weights())
mc_drop.summary()