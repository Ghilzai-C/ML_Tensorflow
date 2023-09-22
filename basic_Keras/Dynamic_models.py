import tensorflow as tf
import shutil
from pathlib import Path
class Wide_and_Deep_model(tf.keras.Model):
    def __init__(self, units=30,activation="relu", **kwargs):
        super().__init__( **kwargs)
        self.norm_layer_wide = tf.keras.layers.Normalization()
        self.norm_layer_deep = tf.keras.layers.Normalization()
        self.hiddedn1 = tf.keras.layers.Dense(units,activation=activation)
        self.hiddedn2 = tf.keras.layers.Dense(units,activation=activation)
        self.main_outPut = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)
    
    def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide = self.norm_layer_wide(input_wide)
        norm_deep = self.norm_layer_deep(input_deep)
        hidden1_layer = self.hiddedn1(norm_deep)
        hidden2_layer = self.hiddedn2(hidden1_layer)
        concat = tf.keras.layers.Concatenate([norm_wide, hidden2_layer])
        output = self.main_outPut(concat)
        aux_output = self.aux_output(hidden2_layer)
        return output, aux_output
tf.random.set_seed(42)
model = Wide_and_Deep_model(30, activation="relu", name="cool_deep_model")

optimizar_o = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", loss_weights=[0.90,0.1],optimizer=optimizar_o,
              metrics="RootMeanSquaredError")
model.norm_layer_wide.adapt(X_train_wide)
model.norm_layer_deep.adapt(X_train_wide)
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=10,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)))

eval_results = model.evaluate((X_test_wide, X_test_deep), (y_test, Y_test))
weigted_sum_of_losses, main_loss, aux_loss,main_rmse = eval_results
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))

# saving the model via shutil
shutil.rmtree("model_saved_via_shutil", ignore_errors= True)
# saving as tf
model.save("model_save_via_tf", save_format="tf")

for path in sorted(Path().glob("model_save_via_tf.*")):
    print(path)

