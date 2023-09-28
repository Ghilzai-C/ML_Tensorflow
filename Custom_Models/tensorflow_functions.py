from custom_loss import*

# def cube(x):
#     return x **3
# print(cube(tf.constant(3)))

# tf_cube = tf.function(cube)
# print(tf_cube(4))

# concrete_function = tf_cube.get_concrete_function(tf.constant(9.0))
# print(concrete_function(tf.constant(3.0)))
# print(concrete_function.graph.get_operations)

# @tf.function
# def tf_cube(x):
#     print(f"x = {x}")
#     return x ** 3
# result  = tf_cube(tf.constant(3.0))
# result
def my_mse(y_true, y_pred):
    print("Tracking the loass my_mse()")
    return tf.reduce_mean(tf.square(y_pred - y_true))
def my_mae(y_true, y_pred):
    print("Tracking metric my_mae()")
    return tf.reduce_mean(tf.abs(y_pred - y_true))

class MyDesnse(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True)
        self.biases = self.add_weight(name='bias',
                                      shape=(self.units),
                                      initializer='zeros',
                                      trainable=True)
        super().build(input_shape)
    
    def call(self, X):
        print("Traking my dense .call()")
        return self.activation(X@ self.kernel + self.biases)
    
tf.random.set_seed(42)

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = MyDesnse(30, activation="relu")
        self.hidden2 = MyDesnse(30, activation="relu")
        self.out_put = MyDesnse(1)
    def call(self, input):
        print("Traking my model .call")
        hidden1 = self.hidden1(input)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([ input, hidden2])
        output = self.out_put(concat)
        return output
my_model = MyModel()

my_model.compile(loss=my_mse,optimizer="nadam", metrics=[my_mae])

# my_model.fit(X_train_scaled, Y_train, epochs=2, validation_data=(X_valid, Y_valid))

# my_model.evaluate(X_test_scaled, Y_test)


class MyMomentumOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.9, name="MyMomentumOptimizer", **kwargs):

        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate)) 
        self._set_hyper("decay", self._initial_decay) 
        self._set_hyper("momentum", momentum)
    
    def _create_slots(self, var_list):

        for var in var_list:
            self.add_slot(var, "momentum")

    @tf.function
    def _resource_apply_dense(self, grad, var):

        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype) 
        momentum_var = self.get_slot(var, "momentum")
        momentum_hyper = self._get_hyper("momentum", var_dtype)
        momentum_var.assign(momentum_var * momentum_hyper - (1. - momentum_hyper)* grad)
        var.assign_add(momentum_var * lr_t)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
        }
tf.random.set_seed(42)
model =tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[8])
])
model.compile(loss="mse", optimizer=MyMomentumOptimizer())
model.fit(X_train_scaled, Y_train, epochs=2)
