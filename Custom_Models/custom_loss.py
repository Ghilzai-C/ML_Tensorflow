import tensorflow as tf 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
def huber_fn(y_true, y_pred):
    error = y_true -y_pred
    is_small_error = tf.abs(error)< 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) -0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

plt.figure(figsize = (8,3.5))
z = np.linspace(-4, 4, 200)
z_center = np.linspace(-1, 1, 200)
plt.plot(z, huber_fn(0, z), "b-", linewidth=2, label="huber($z$)")
plt.plot(z, z**2/2, "r:", linewidth=1)
plt.plot(z_center, z_center**2 / 2, "r", linewidth = 2)
plt.plot([-1, 1], [0, huber_fn(0., -1.)], "k--")
plt.plot([1, 1],[0, huber_fn(0., 1.)], "k--")
plt.gca().axhline(y=0, color='k')
plt.gca().axvline(x=0, color='k')
plt.text(2.1, 3.5, r"$\frac{1}{2}z^2$", color = "r", fontsize= 15)
plt.text(3.0, 2.2, r"$|z| - \frac{1}{2}$", color = "b", fontsize= 15)
plt.axis([-4, 4, 0, 4])
plt.grid(True)
plt.xlabel("$z$")
plt.legend(fontsize=14)
plt.title("huber_loss", fontsize = 14)
# plt.show()

housing = fetch_california_housing()
X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    housing.data, housing.target.reshape(-1,1), random_state=42
)
X_train, X_valid, Y_train, Y_valid= train_test_split(
    X_train_full, Y_train_full, random_state= 42
)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.fit_transform(X_valid)
X_test_scaled = scaler.fit_transform(X_test)
input_shape = X_train.shape[1:]
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu", kernel_initializer="he_normal",
                          input_shape = input_shape),
    tf.keras.layers.Dense(1)
])
model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])
# history_custom_huber = model.fit(X_train_scaled, Y_train, epochs=2,
#                                  validation_data=(X_valid, Y_valid))
# model.save("Model_with_custom_loss")

# loded_model = tf.keras.models.load_model("Model_with_custom_loss", 
                                        #  custom_objects={"huber_fn":huber_fn})
# loded_model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])
# loded_model.fit(X_train_scaled, Y_train, epochs=2,
#                 validation_data= (X_valid_scaled, Y_valid))

def create_huber(threshold = 1.0):
    def huber_fn(y_true, y_pred):
        error = y_true -y_pred
        is_small_error = tf.abs(error)< threshold
        squared_loss = tf.square(error) / 2
        linear_loss =  threshold * tf.abs(error) - threshold ** 2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn
# advance_model = tf.keras.models.load_model("Model_with_custom_loss")
# advance_model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=["mae"])
# advance_model.fit(X_train_scaled, Y_train, epochs=2,
#                 validation_data= (X_valid_scaled, Y_valid))
class HuberLoss(tf.keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_error = self.threshold * tf.abs(error) - self.threshold** 2 / 2
        return tf.where(is_small_error, squared_loss, linear_error)
    def get_conf(self):
        base_config = super().get_conf()
        return {**base_config, "threshold":self.threshold}
# advance_model_2 = tf.keras.models.load_model("Model_with_custom_loss", custom_objects=
                                            #  {"huber_fn": create_huber(2.0)})
# advance_model_2.compile(loss=HuberLoss(2.0), optimizer="nadam", metrics=["mae"])

# advance_model_2.fit(X_train_scaled, Y_train, epochs=2,
#                 validation_data= (X_valid_scaled, Y_valid))


# Additional functions 

def my_softplus(z):
    return tf.math.log(1.0 + tf.exp(z))
def my_glorot_initilizer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0]+ shape[1]))
    return tf.random.normal(shape,stddev= stddev, dtype=dtype)
def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 *weights))
def my_positive_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)


my_layer = tf.keras.layers.Dense(1 , activation=my_softplus,
                              kernel_initializer=my_glorot_initilizer,
                              kernel_regularizer=my_l1_regularizer,
                              kernel_constraint=my_positive_weights)
tf.random.set_seed(42)

my_model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu", kernel_initializer="he_normal",
                          input_shape= input_shape),
    my_layer
])
my_model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
# my_model.fit(X_train_scaled, Y_train, epochs=2,
#              validation_data=(X_valid_scaled, Y_valid))
# my_model.save("model_with_custom_kernel")

# loaded_model1 = tf.keras.models.load_model("model_with_custom_kernel",
#                                            custom_objects={
#                                                "my_l1_regularizer":my_l1_regularizer,
#                                                "my_positive_weights":my_positive_weights,
#                                                "my_glorot_initializer":my_glorot_initilizer,
#                                                "my_softplus":my_softplus
#                                            })
# loaded_model1.fit(X_train_scaled, Y_train, epochs=2,
#                   validation_data = (X_valid_scaled, Y_valid))

class MyL1Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    def get_config(self):
        return {"factor":self.factor}