from faster_optimizers import*

import math
import numpy as np
batch_size = 32
n_epochs = 25
n_steps = n_epochs * math.ceil(len(X_train)/ batch_size)
n_steps_per_epoch = math.ceil(len(X_train) / batch_size)
scheduled_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01, decay_steps=n_steps, decay_rate=0.1
)
optimizer_ker_sch = tf.keras.optimizers.SGD(learning_rate=  scheduled_learning_rate)

# model = build_and_train_model(optimizer_ker_sch)
# time  27.29 acc   88.29

piece_scheduled_learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[5.0*n_steps_per_epoch, 15.*n_steps_per_epoch],
    values=[0.01, 0.005, 0.001]
)
optimizer_ker_piec = tf.keras.optimizers.SGD(learning_rate=piece_scheduled_learning_rate)
model_ker_piec= build_and_train_model(optimizer_ker_piec)
# time 31   acc 88.44

end_time = timeit.default_timer() - start_time
print(end_time)