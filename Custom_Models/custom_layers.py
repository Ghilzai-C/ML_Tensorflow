from custom_metrics import*
exponential_layer = tf.keras.layers.Lambda(lambda x: tf.exp(x))
print(exponential_layer([-1., 0., 1.]))

tf.random.set_seed(42)

model_layer = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu",input_shape=input_shape),
    tf.keras.layers.Dense(1),
    exponential_layer
])
model_layer.compile(loss="mse", optimizer="sgd")
# model_layer.fit(X_train_scaled, Y_train, epochs=5,
#                 validation_data=(X_valid,Y_valid))
# model_layer.evaluate(X_test_scaled, Y_test)

class MyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units 
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self.units],
            initializer="he_normal"
        )
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros"
        )
        super().build(batch_input_shape)
    def call(self, x):
        return self.activation(x @ self.kernel + self.bias)
    
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])
    def get_config(self):
        base_config= super().get_config()
        return {**base_config, "units":self.units,
                "activation":tf.keras.activations.serialize(self.activation)}
        
tf.random.set_seed(42)
model_custom_dense = tf.keras.Sequential([
    MyDense(30, activation="relu",input_shape=input_shape),
    MyDense(1)
])

model_custom_dense.compile(loss="mse", optimizer="nadam")
# model_custom_dense.fit(X_train_scaled, Y_train, epochs=2,
#                        validation_data=(X_valid, Y_valid))
# model_custom_dense.evaluate(X_test_scaled, Y_test)

class MultiLayer(tf.keras.layers.Layer):
    def call(self, x):
        x1, x2 = x
        print("x1.Shape", x1.shape, "x2.Shape", x2.shape)
        return x1+ x2, x1*x2, x1/x2
    def compute_output_shape(self, batch_input_shape):
        batch_input_shape1, batch_input_shape2 = batch_input_shape
        return [batch_input_shape1, batch_input_shape1, batch_input_shape1]
input1 = tf.keras.layers.Input(shape=[2])
input2 = tf.keras.layers.Input(shape=[2])
MultiLayer()((input1, input2))
