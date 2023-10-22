import tensorflow as tf 
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes

print (dataset_size)
print(class_names)
print(n_classes)

test_set_raw, valid_set_raw, train_set_raw = tfds.load("tf_flowers",
                                                       split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
                                                       as_supervised=True)

# plt.figure(figsize=(12,10))
# index = 0

# for image, lable in valid_set_raw.take(9):
#     index += 1
#     plt.subplot(3,3,index)
#     plt.imshow(image)
#     plt.title(f"Class:{class_names[lable]}")
#     plt.axis("off")
# plt.show()

tf.keras.backend.clear_session()

batch_size = 32
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=224,width=224, crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])
train_set = train_set_raw.map(lambda X, y:(preprocess(X), y))
train_set = train_set.shuffle(1000, seed = 42).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)
test_set = test_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)

# plt.figure(figsize=(12,12))
# for X_batch, Y_batch in valid_set.take(1):
#     for index in range(9):
#         plt.subplot(3,3,index+1)
#         plt.imshow((X_batch[index]+1) / 2)
#         plt.title(f"Class:{class_names[Y_batch[index]]}")
#         plt.axis("off")
# plt.show()

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
    tf.keras.layers.RandomRotation(factor=0.05, seed= 42),
    tf.keras.layers.RandomContrast(factor=0.2, seed= 42)
])

# plt.figure(figsize=(12,12))
# for X_batch, Y_batch in valid_set.take(1):
#     X_batch_augmented = data_augmentation(X_batch, training=True)
#     for index in range (9):
#         plt.subplot(3,3, index+1)
#         plt.imshow(np.clip((X_batch_augmented[index]+1)/ 2, 0, 1))
#         plt.title(f"Class: {class_names[Y_batch[index]]}")
#         plt.axis("off")
# plt.show()
        
tf.random.set_seed(42)
bas_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(bas_model.output)
output= tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
t_model = tf.keras.Model(inputs=bas_model.input, outputs = output)
t_model.summary()

# for layer in bas_model.layers:
#     layer.trainable = False
    
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
t_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,metrics=["accuracy"])
# t_history = t_model.fit(train_set, validation_data=valid_set, epochs=3)

for indices in zip(range(33), range(33,66), range(66, 99), range(99, 132)):
    for idx in indices:
        print(f"{idx:3} : {bas_model.layers[idx].name:22}", end="")
    print()


for layer in bas_model.layers[56:]:
    layer.trainable =True

ft_history = t_model.fit(train_set, validation_data=valid_set, epochs=10)
