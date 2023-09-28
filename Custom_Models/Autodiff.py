from custom_loss import*
# import math 
# def f(w1, w2):
#     return  3 * w1 ** 2 + 2 * w1 *w2 

# w1 , w2 = tf.Variable(5.), tf.Variable(3.)
# with tf.GradientTape() as tape:
#     z = f(w1, w2)

# gradients = tape.gradient(z, [w1, w2])
# print(gradients)
# x = tf.Variable(1e-50)
# with tf.GradientTape(persistent=True) as tape:
#     z = f(w1, w2)
#     z1 = f(w1, w2+5.)
#     z2 = tf.sqrt(x)
# dz_dw1 = tape.gradient(z, w2)
# print(dz_dw1)
# s_dw = tape.gradient([z, z1], [w1, w2])
# print(s_dw)
# print(tape.gradient(z, [x]))

# def soft_plus(z):
#     return tf.math.log(1 + tf.exp(-tf.abs(z))) + tf.maximum(0., z)
# x = tf.Variable([1.0e30])
# with tf.GradientTape() as tape:
#     z = soft_plus(x)
# print(tape.gradient(z, [x]))

tf.random.set_seed(42)

l2_reg = tf.keras.regularizers.l2(0.05)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, activation="relu", kernel_initializer="he_normal",
                          kernel_regularizer= l2_reg),
    tf.keras.layers.Dense(1, kernel_regularizer=l2_reg)
])
def random_batch(x, y, batch_size=32):
    idx = np.random.randint(len(x),size= batch_size)
    return x[idx], y[idx]
def print_status(step, total, loss, metrics=None):
    metrics= " - ".join([f"{m.name}:{m.result():.4f}"
                        for m in [loss] + (metrics or [])])
    end = " - "if step < total else "\n"
    print(f"\r{step}/{total} - " + metrics, end=end)
np.random.seed(42)
tf.random.set_seed(42)

n_epochs = 5
batch_size = 32
n_step = len(X_train) // batch_size
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.mean_squared_error
mean_loss = tf.keras.metrics.Mean()
metrics = [tf.keras.metrics.MeanAbsoluteError()]
# for epoch in range(1, n_epochs + 1):
#     print(f"Epoch{epoch}/{n_epochs}")
#     for step in range(1, n_step +1):
#         x_batch, y_batch = random_batch(X_train_scaled, Y_train)
#         with tf.GradientTape() as tape:
#             y_pred = model(x_batch, training= True)
#             main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
#             loss = tf.add_n([main_loss]+model.losses)
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         for variable in model.variables:
#             if variable.constraint is not None:
#                 variable.assign(variable.constraint(variable))
#         mean_loss(loss)
#         for metric in metrics:
#             metric(y_batch, y_pred)
#         print_status(step, n_step, mean_loss, metrics)
#     for metric in [mean_loss]+ metrics:
#         metric.reset_states()
from tqdm.notebook import trange
from collections import OrderedDict

with trange(1, n_epochs+1, desc="All epochs") as epochs:
    for epoch in epochs:
 
        with trange(1, n_steps + 1, desc=f"Epoch{epoch}/{n_epochs}") as steps:
            for step in steps:
                x_batch, y_batch = random_batch(X_test_scaled, Y_train)
                with tf.GradientTape() as tape:
                    y_pred = model(x_batch)
                    main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                    loss = tf.add_n([main_loss]+ model.losses)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                for variable in model.variables:
                    if variable.constraint is not None:
                        variable.assign(variable.constraint(variable))
                status = OrderedDict()
                mean_loss(loss)
                status["loss"] = mean_loss.result().numpy()
                for metric in metrics:
                    metric(y_batch, y_pred)
                    status[metric.name] - metric.result().numpy()
                steps.set_postfix(status)
        for metric in [mean_loss] + metrics:
            metric.reset_states()
            
        

