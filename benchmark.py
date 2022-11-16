# Created by David at 24.10.2022
# Project name sdu_cat_dog

import tensorflow as tf
from keras import Sequential
from tensorflow import keras



class_names = ["dog","cat"]
NUM_CLASSES = 2
image_size = (512, 512)
batch_size = 64

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/catdog_data/catdog_data/test",
    labels="inferred",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/catdog_data/catdog_data/validation",
    labels="inferred",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

test_ds = test_ds.prefetch(buffer_size=32)

def make_model(input_shape, num_classes):
    model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=1.0,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=num_classes,
        classifier_activation="softmax",
    )
    return model

model = make_model(input_shape=image_size + (3,), num_classes=2)
#keras.utils.plot_model(model, show_shapes=True)




test_ds = test_ds.map(lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)))
#train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)))
validation_ds = validation_ds.map(lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)))

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
tf.get_logger().setLevel('INFO')

model.load_weights("models/save_at_3.h5")
print("Test dataset")
model.evaluate(test_ds)
print("validation_ds dataset")
model.evaluate(validation_ds)