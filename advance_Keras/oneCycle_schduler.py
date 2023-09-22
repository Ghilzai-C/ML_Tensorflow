from faster_optimizers import*
import numpy as np 
import math 


K = tf.keras.backend

class ExponentialLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_epoch_begin(self, epocj, logs= None):
        self.sum_of_epoch_losses = 0
    
    def on_batch_end(self, batch, logs=None):
        mean_epoch_loss = logs["loss"]
        new_sum_of_epochs_loss = mean_epoch_loss * (batch +1)
        batch_loss = new_sum_of_epochs_loss - self.sum_of_epoch_losses
        self.sum_of_epoch_losses = new_sum_of_epochs_loss
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(batch_loss)
        K.set_value(self.model.optimizer.learning_rate,
                    self.model.optimizer.learning_rate * self.factor)
        
def find_learning_rate(model, X, Y, epochs = 1, batch_size=32, min_rate=1e-4,
                       max_rate=1):
    init_weights = model.get_weights()
    itrations = math.ceil(len(X)/ batch_size) * epochs
    factor = (max_rate/ min_rate) **(1/itrations)
    init_lr = K.get_value(model.optimizer.learning_rate)
    K.set_value(model.optimizer.learning_rate, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, Y, epochs=epochs, batch_size =batch_size,
                        callbacks=[exp_lr])
    
    K.set_value(model.optimizer.learning_rate, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates , exp_lr.losses
model = build_model()
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.SGD(learning_rate=  0.001),
              metrics=["accuracy"])
def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses  , "b")
    plt.gca().set_xscale('log')
    max_loss = losses[0] +min(losses)
    plt.hlines(min(losses), min(rates), max(rates), color = 'k')
    plt.axis([min(rates), max(rates), 0, max_loss])
    plt.xlabel("learning_rate")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

batch_size = 128
# rates, losses = find_learning_rate(model, X_train, Y_train, epochs=1, batch_size=batch_size)
# plot_lr_vs_loss(rates, losses)

class OneCycleSchduler(tf.keras.callbacks.Callback):
    def __init__(self, iterations, max_lr=1e-3, start_lr=None,
                 last_iterations=None, last_lr=None):
        self.iterations = iterations
        self.max_lr = max_lr
        self.start_lr = start_lr or max_lr / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_lr = last_lr or self.start_lr / 1000
        self.iteration = 0
    
    def _interpolate(self, iter1, iter2, lr1, lr2):
        return(lr2-lr1)*(self.iteration -iter1)/ (iter2 - iter1) + lr1
    
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            lr = self._interpolate(0, self.half_iteration, self.start_lr, self.max_lr)
        elif self.iteration < 2 * self.half_iteration:
            lr = self._interpolate(self.half_iteration, 2*self.half_iteration, 
                                   self.max_lr, self.last_lr)
        else:
            lr = self._interpolate(2* self.half_iteration, self.iterations, 
                                   self.start_lr, self.last_lr)
        self.iteration += 1
        K.set_value(self.model.optimizer.learning_rate, lr)

n_epochs = 25
oneCycle = OneCycleSchduler(math.ceil(len(X_train) / batch_size) * n_epochs,
                             max_lr=0.1)
history_oneCycle = model.fit(X_train, Y_train, epochs=n_epochs, batch_size=batch_size,
                             validation_data=(X_valid, Y_valid),
                             callbacks=[oneCycle])


# time 21.59 acc 92.63
end_time = timeit.default_timer() - start_time
print(end_time)