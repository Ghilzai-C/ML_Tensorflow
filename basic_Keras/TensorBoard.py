from pathlib import Path
import tensorboard
import tensorflow as tf 
from time import strftime
import numpy as np

def get_run_logDir(root_logdir="my_log"):
    return Path(root_logdir)/ strftime("run_%y_%m_%d_%H_%M_%S")
run_logDir = get_run_logDir()

tf.keras.backend.clear_session()
tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(30, activation="relu")
    tf.keras.layers.Dense(30, activation="relu")
    tf.keras.layers.Dense(1)
])

optmizer_o = tf.keras.optimizers.SGD(learning_rate=1e-3)
model.compile(loss="mse",optimizer=optmizer_o, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)

tensorBoard_cb = tf.keras.callbacks.TensorBoard(run_logDir, profile_batch=(100, 200))

history = model.fit((X_train, Y_train, epochs=3,
                     validation_data=(X_valid, Y_valid),
                     callbacks=[tensorBoard_cb]))
print("my_logs")
for path in sorted(Path("my_logs").glob("**/*")):
    print(" " * (len(path.parts) - 1)+path.parts[-1])

# for opening tensorboard in web 
from IPython.display import display, HTML
display(HTML('<a href="http://localhost:6006/">http://localhost:6006/</a>'))
test_logdir = get_run_logDir()
writer = tf.summary.create_file_writer(str(test_logdir))
with writer.as_default():
    for step in range(1, 1000+1):
        tf.summary.scalar("my_scaler", np.sin(step / 10), step=step)
        
        data = (np.random.rand(100)+2) *step /100
        tf.summary.histogram("my_hist",data, buckets=50, step=step)
        
        images = np.random.rand(2, 32, 32, 3) *step /1000
        tf.summary.image("my_images", images,step=step)
        
        tests = ["the step is "+ str(step), "its square is "+ str(step**2)]
        tf.summary.text("my_text", tests, step=step)
        
        sine_wave = tf.math.sin(tf.range(12000)/48000 * 2 * np.pi *step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
        tf.summary.audio("my_audio", audio, step=step)
        
        