from mnist_data import*
from functools import partial
from sklearn.datasets import load_sample_images
import matplotlib.pyplot as plt 


DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1, padding="same",
                        kernel_initializer="he_normal", use_bias=False)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides = 1, activation= "relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()
        ]
        
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()
            ]
    def call (self, inputs):
        z = inputs
        for layer in self.main_layers:
            z = layer(z)
        skip_z = inputs
        for layer in self.skip_layers:
            skip_z = layer(skip_z)
        return self.activation(z + skip_z)
model = tf.keras.Sequential([
    DefaultConv2D(64, kernel_size=7, strides=2, input_shape = [224,224,3]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
])
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters ==prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))


model.compile(loss = "sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model.summary()


# pretrained model 
model_pretrain = tf.keras.applications.ResNet50(weights="imagenet")

images = load_sample_images()["images"]

images_resized = tf.keras.layers.Resizing(height=224, width=224,
                                          crop_to_aspect_ratio= True)(images)

inputs = tf.keras.applications.resnet50.preprocess_input(images_resized)
y_proba = model_pretrain.predict(inputs)
print(y_proba.shape)

top_k = tf.keras.applications.resnet50.decode_predictions(y_proba, top =3)
for image_index in range(len(images)):
    print(f"image # {image_index}")
    for class_id, name, y_proba in top_k[image_index]:
        print(f" {class_id} - {name:12s}  {y_proba:.2%}")

plt.figure(figsize=(10, 6))
for idx in (0,1):
    plt.subplot(1,2,idx +1)
    plt.imshow(images_resized[idx]/255)
    plt.axis("off")
plt.show()