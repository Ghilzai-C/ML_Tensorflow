import tensorflow as tf 
import keras_tuner as kt 
import tensorboard
from pathlib import Path
import timeit

current_time = timeit.default_timer

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()

(X_train_full, Y_train_full), (X_test, Y_test) = fashion_mnist
X_train, Y_train = X_train_full[:-5000], Y_train_full[:-5000]
X_valid, Y_valid = X_train_full[-5000:], Y_train_full[-5000:]

tf.keras.backend.clear_session()
tf.random.set_seed(42)

def build_model(hp):
    n_hidden =hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    
    optmizer = hp.Choice("optmizer", values=["sgd","adam"])
    if optmizer == "sgd":
        optmizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optmizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model= tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons,activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optmizer,
                  metrics=["accuracy"])
    return model

random_search_tuner = kt.RandomSearch(
    build_model, objective="val_accuracy", max_trials=5, overwrite=True,
    directory="my_earth_quake_data", project_name="earthquake_rnd_search", seed=42
)
random_search_tuner.search(X_train, Y_train, epochs=10,validation_data=(X_valid, Y_valid))


top3_models = random_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]
top3_parameters = random_search_tuner.get_best_hyperparameters(num_trials=3)
top3_parameters[0].values

best_trail = random_search_tuner.oracle.get_best_trials(num_trials=1)[0]
best_trail.summary    

best_trail.metrics.get_last_value("val_accuracy")
best_model.fit(X_train_full, Y_train_full, epochs=10)
test_loss, test_accuracy = best_model.evaluate(X_test, Y_test)

class MyClassificationHyperModel(kt.HyperModel):
    def build(self, hp):
        return build_model(hp)

    def fit(self, hp, model, X, Y, **kwargs):
        if hp.Boolean("normalize"):
            norm_layer = tf.keras.layers.Normalization()
            X = norm_layer(X)
        return model.fit(X, Y, **kwargs)
hyperband_tuner = kt.Hyperband(
    MyClassificationHyperModel(), objective="val_accuracy", seed=42, max_epochs=10,
    factor=3, hyperband_iterations=2,
    overwrite=True, directory = "my_earthquake_data", project_name="hyperband"
)

root_logdir = Path(hyperband_tuner.project_dir) / "tensorboard"
tensorboard_cb = tf.keras.callbacks.TensorBoard(root_logdir)
early_stoping_cb = tf.keras.callbacks.EarlyStopping(patience = 2)
hyperband_tuner.search(X_train, Y_train, epochs=10,
                       validation_data=(X_valid,Y_valid),
                       callbacks=[early_stoping_cb, tensorboard_cb])



bayesian_opt_tuner = kt.BayesianOptimization(
    MyClassificationHyperModel(), objective="val_accuracy", seed=42,
    max_trials=10, alpha=1e-4, beta=2.6,
    overwrite=True, directory="my_fashion_mnist", project_name="bayesian_opt")
bayesian_opt_tuner.search(X_train, Y_train, epochs=15,
                          validation_data=(X_valid, Y_valid),
                          callbacks=[early_stoping_cb])

finalTime = current_time - timeit.default_timer
finalTime /= 60
print(finalTime)