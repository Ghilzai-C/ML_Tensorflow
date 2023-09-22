import tensorflow as tf 

tf.keras.backend.clear_session()
tf.random.set_seed(42)

normalization_layer = tf.keras.layers.Normalization()
hiddenlayer1 = tf.keras.layers.Dense(30, activation="relu")
hiddenlayer2 = tf.keras.layers.Dense(30, activation="relu")
contact_layer = tf.keras.layers.Concatenate()
outPut_layer = tf.keras.layers.Dense(1)
input_ = tf.keras.layers.Input(shape=X_train.shape[1:])
normalized = normalization_layer(input_)
hidden1 = hiddenlayer1(normalized)
hidden2 = hiddenlayer2(hidden1)
concat = contact_layer(normalized, hidden2)
output = outPut_layer(concat)
model = tf.keras.Model(inputs=[input_], outputs=[output])
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse",optimizer=optimizer, metrics=["RootMeanSquaredError"])
normalization_layer.adapt(X_train)
history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid))
mse_test = model.evaluate(X_test, Y_test)
y_pred = model.predict(X_new)

# for different subset input features 
tf.random.set_seed(42)

input_wide = tf.keras.layers.Input(shape=[5])
input_deep = tf.keras.layers.Input(shape=[6])
norm_wide_layer = tf.keras.layers.Normalization()
norm_deep_layer = tf.keras.layers.Normalization()
norm_wide = norm_wide_layer(input_wide)
norm_deep = norm_deep_layer(input_deep)
hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")(hidden_layer1)
concat2 = tf.keras.layers.Concatenate([norm_wide, hidden_layer2])
output2 = tf.keras.layers.Dense(1)(concat2)
model2 = tf.keras.Model(inputs=[input_wide, input_deep], outputs = [output2])

optimizer2 = tf.keras.optimizers.Adam(learning_rate=1e-3)
model2.compile(loss="mse",optimizer=optimizer2, metrics=["RootMeanSquareError"])
X_train_wide, X_train_deep= X_train[:,:5], X_train[:,:2]
X_valid_wide, X_valid_deep = X_valid[:,:5], X_valid[:,:2]
X_test_wide, X_test_deep = X_test[:,:5], X_test[:,:2]
X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]

norm_wide_layer.adapt(X_train_wide)
norm_deep_layer.adapt(X_train_deep)
history = model2.fit((X_train_wide, X_train_deep), Y_train, epochs = 20,
                     validation_data= ((X_valid_wide, X_valid_deep), Y_valid))
mse_test2 = model2.evaluate((X_test_wide, X_test_deep), Y_test)
y_pred = model2.predict((X_new_wide, X_new_deep))

# model for dual auxiliary output 

input_wide = tf.keras.layers.Input(shape=[5])
input_deep = tf.keras.layers.Input(shape=[6])
norm_wide_layer = tf.keras.layers.Normalization()
norm_deep_layer = tf.keras.layers.Normalization()
norm_wide = norm_wide_layer(input_wide)
norm_deep = norm_deep_layer(input_deep)
hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")(hidden_layer1)
concat2 = tf.keras.layers.Concatenate([norm_wide, hidden_layer2])
output2 = tf.keras.layers.Dense(1)(concat2)
aux_output = tf.keras.layers.Dense(1)(hidden_layer2)
model3 = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output2, aux_output])

optimizer3 = tf.keras.optimizers.Adam(learning_rate=1e-3)
model3.compile(loss=("mse","mse"), loss_weights=(0.9, 0.1), optimizer=optimizer3,
               metrics=["RootMeanSquaredError"])

norm_wide_layer.adapt(X_train_wide)
norm_deep_layer.adapt(X_train_deep)
history = model2.fit((X_train_wide, X_train_deep), Y_train, epochs = 20,
                     validation_data= ((X_valid_wide, X_valid_deep), Y_valid))

eval_results = model3.evaluate((X_test_wide, X_test_deep),(Y_test, Y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse = eval_results
y_predict_main, y_predict_aux = model3.predict( X_new_wide, X_new_deep)
y_predict_tuple = model3.predict((X_new_wide, X_new_deep))
