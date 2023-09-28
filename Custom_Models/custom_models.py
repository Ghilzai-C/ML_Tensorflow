from custom_loss import*

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu",
                                             kernel_initializer="he_normal")
                       for _ in range(n_layers)]
    def call(self, inputs):
        z = inputs
        for layer in self.hidden:
            z = layer(z)
        return inputs + z

class Residual_regressor(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = tf.keras.layers.Dense(30, activation="relu",
                                             kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = tf.keras.layers.Dense(output_dim)
    def call(self, inputs):
        z = self.hidden1(inputs)
        for _ in range(1 + 3):
            z = self.block1(z)
        z = self.block2(z)
        return self.out(z)
tf.random.set_seed(42)

model_res = Residual_regressor(1)
model_res.compile(loss="mse", optimizer="nadam")
# history_rse = model_res.fit(X_train_scaled, Y_train, epochs=2, validation_data=(X_valid_scaled, Y_valid))
# score = model_res.evaluate(X_test_scaled, Y_test)

class ReconstructingRegressor(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Dense(30, activation="relu",
                                             kernel_initializer="he_normal")
                       for _ in range(5)]
        self.out = tf.keras.layers.Dense(output_dim)
        self.reconstruction_mean = tf.keras.metrics.Mean(
            name = "reconstruction_error"
        )
    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = tf.keras.layers.Dense(n_inputs)
        super().build(batch_input_shape)
    def call(self, inputs, training = None):
        z = inputs
        for layer in self.hidden:
            z = layer(z)
        reconstruction= self.reconstruct(z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        if training:
            result = self.reconstruction_mean(recon_loss)
            self.add_metric(result)
        return self.out(z)
    
tf.random.set_seed(42)
model_reconst = ReconstructingRegressor (1)
model_reconst. compile(loss="mse", optimizer="nadam")
model_reconst.fit(X_train_scaled, Y_train, epochs=5)
model_reconst.predict(X_test_scaled)


