import tensorflow as tf 
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_saved_via_tf", save_weights_only=True)
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=10,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)),
    callbacks=[checkpoint_cb])

early_stoping_cb = tf.keras.callbacks.EarlyStopping("model_saved_via_tf", restore_best_weights=True)

history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=10,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)),
    callbacks=[checkpoint_cb, early_stoping_cb])

class PrintValTrainingRatioCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        ratio = logs["val_loss"]/logs["loss"]
        print(f"Epoch = {epoch}, val/train = {ratio:.2f}")
val_train_ratio_cb = PrintValTrainingRatioCallBack()
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=10,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)),
    callbacks=[val_train_ratio_cb], verbose=0)
