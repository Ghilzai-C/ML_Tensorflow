from faster_optimizers import*
import math
import numpy as np

tf.random.set_seed(42)
optimizer_power = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-4)

# history_power_sch = build_and_train_model(optimizer_power)

# time 26.9 acc 88.04

learning_rate = 0.01
decay = 1e-4
batch_size = 32
n_steps_per_epoch = math.ceil(len(X_train) / batch_size)
n_epochs = 25
# epochs = np.arange(n_epochs)

# lrs = learning_rate / (1 + decay*epochs * n_steps_per_epoch)

# plt.plot(epochs, lrs, "o-")

# plt.axis([0, n_epochs - 1, 0, 0.01])
# plt.xlabel("epoch")
# plt.ylabel("learning rate")
# plt.title("power_schuduling", fontsize = 14)
# plt.grid(True)
# plt.show()



def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn
exponential_decay_fn = exponential_decay(lr0= 0.01, s = 20)
model = build_model()
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
# model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
#               metrics=["accuracy"])

lr_scheduler =tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
# history_lr_sch = model.fit(X_train, Y_train, epochs=n_epochs,
#                            validation_data=(X_valid, Y_valid),
#                            callbacks=[lr_scheduler])
# time 66.0, acc 89.61


K = tf.keras.backend
class  ExponentialDecay(tf.keras.callbacks.Callback):
    def __init__(self, n_steps = 40_000):
        super().__init__()
        self.n_steps = n_steps
    def on_batch_begin(self, batch, logs = None):
        lr = K.get_value(self.model.optimizer.learning_rate)
        new_learning_rate = lr * 0.1 **( 1 / self.n_steps)
        K.set_value(self.model.optimizer.learning_rate, new_learning_rate)
        
    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.learning_rate)
        
lr0 = 0.01
optimizer = tf.keras.optimizers.SGD(learning_rate=lr0)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])

n_steps = n_epochs * math.ceil(len(X_train)/ batch_size)
esp_decay = ExponentialDecay(n_steps)
# history_esp_dec = model.fit(X_train, Y_train, epochs=n_epochs,
#                            validation_data=(X_valid, Y_valid),
#                            callbacks=[esp_decay])
# time 90   acc 89.8


# Piecewise Constant Scheduling
# def piecewise_constant_fn(epoch):
#     if epoch < 5:
#         return 0.01
#     if epoch < 15:
#         return 0.005
#     else:
#         return 0.001
def pieceWise_constant(boundary, value):
    boundary = np.array([0] + boundary)
    value = np.array(value)
    def piecewise_constant_fn(epoch):
        return value[(boundary > epoch).argmax() -1]
    return piecewise_constant_fn
pieceWise_constant_fn = pieceWise_constant([5,15], [0.01, 0.005, 0.001])
lr_piecew_sch = tf.keras.callbacks.LearningRateScheduler(pieceWise_constant_fn)
optmizer_piW_sch = tf.keras.optimizers.Nadam(learning_rate=lr0)
# history_piecewise_sch = model.fit(X_train, Y_train, epochs=n_epochs,
#                                   validation_data=(X_valid, Y_valid),
#                                   callbacks=[lr_piecew_sch])
# time 64.25   acc 90.15
# plt.plot(history_piecewise_sch.epoch, history_piecewise_sch.history["lr"],"o-")
# plt.axis([0, n_epochs - 1, 0, 0.011])
# plt.xlabel("Epoch")
# plt.ylabel("Learning rate")
# plt.title("PieceWise Constant Scheduling ", fontsize= 14)
# plt.grid(True)
# plt.show()

# Performance Scheduling 
optmizer_performance = tf.keras.optimizers.SGD(learning_rate=lr0)
model.compile(loss="sparse_categorical_crossentropy", optimizer= optmizer_performance,
              metrics=["accuracy"])
lr_pr_sch = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=5)
history_pr_sch = model.fit(X_train, Y_train, epochs=n_epochs,
                           validation_data=(X_valid, Y_valid), 
                           callbacks=[lr_pr_sch])
# time 66.01   acc 91.64
plt.plot(history_pr_sch.epoch, history_pr_sch.history["lr"], "bo-")
plt.xlabel("Epochs")
plt.ylabel("learning rate", color = 'b')
plt.tick_params('y', color = 'b')
plt.gca().set_xlim(0, n_epochs-1)
plt.grid(True)

ax2 = plt.gca().twinx()
ax2.plot(history_pr_sch.epoch, history_pr_sch.history["val_loss"], "r^-")
ax2.set_ylabel('validation loss', color='r')
ax2.tick_params('y', color='r')
plt.title("reduce LR on Plateau", fontsize = 14)
plt.show()
















end_time = timeit.default_timer() - start_time
print(end_time)