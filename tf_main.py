# Created by David at 24.10.2022
# Project name sdu_cat_dog

import tensorflow as tf
from keras import Sequential
from tensorflow import keras
from tensorflow.keras import layers

class_names = ["dog","cat"]

image_size = (512, 512)
batch_size = 8

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/catdog_data/catdog_data/train_augmentation",
    labels="inferred",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/catdog_data/catdog_data/validation",
    labels="inferred",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

def make_model(input_shape, num_classes):
    model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=1.0,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=1,
        classifier_activation="softmax",
    )
    return model

model = make_model(input_shape=image_size + (3,), num_classes=2)
#keras.utils.plot_model(model, show_shapes=True)

epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
tf.get_logger().setLevel('INFO')
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)