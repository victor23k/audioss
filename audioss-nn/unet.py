import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

INPUT_HEIGHT = 512
INPUT_WIDTH = 256
KERNEL_SIZE = (3,3)
LEAKINESS = 0.2

def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=LEAKINESS)

def UNet_model():

    inputs = keras.Input(shape=(512, 256, 1), name="inputs")
    x = layers.Conv2D(filters=64, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="e_conv1_1")(inputs)
    x = layers.Conv2D(filters=64, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="e_conv1_2")(x)
    x = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same', name="e_pool1")(x)
    x = layers.BatchNormalization(name="e_batch_norm1")(x)

    x = layers.Conv2D(filters=128, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="e_conv2_1")(x)
    x = layers.Conv2D(filters=128, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="e_conv2_2")(x)
    # x = layers.Conv2D(filters=128, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="e_conv2_3")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name="e_pool2")(x)
    x = layers.BatchNormalization(name="e_batch_norm2")(x)

    x = layers.Conv2D(filters=256, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="e_conv3_1")(x)
    x = layers.Conv2D(filters=256, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="e_conv3_2")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name="e_pool3")(x)
    x = layers.BatchNormalization(name="e_batch_norm3")(x)

    x = layers.Conv2D(filters=512, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="e_conv4_1")(x)
    x = layers.Conv2D(filters=512, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="e_conv4_2")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name="e_pool4")(x)
    x = layers.BatchNormalization(name="e_batch_norm4")(x)

    x = layers.Conv2D(filters=1024, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="e_conv5_1")(x)
    x = layers.Conv2D(filters=1024, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="e_conv5_2")(x)

    x = layers.Conv2DTranspose(filters=512, kernel_size=(2,2), strides=2, activation=tf.nn.relu, padding='same', name="d_tconv0")(x)

    x = layers.Conv2D(filters=256, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="d_conv1_1")(x)
    x = layers.Conv2D(filters=256, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="d_conv1_2")(x)
    x = layers.Conv2DTranspose(filters=256, kernel_size=(2,2), strides=2, activation=tf.nn.relu, padding='same', name="d_tconv1")(x)
    x = layers.BatchNormalization(name="d_batch_norm1")(x)

    x = layers.Conv2D(filters=128, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="d_conv2_1")(x)
    x = layers.Conv2D(filters=128, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="d_conv2_2")(x)
    x = layers.Conv2DTranspose(filters=128, kernel_size=(2,2), strides=2, activation=tf.nn.relu, padding='same', name="d_tconv2")(x)
    x = layers.BatchNormalization(name="d_batch_norm2")(x)

    x = layers.Conv2D(filters=64, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="d_conv3_1")(x)
    x = layers.Conv2D(filters=64, kernel_size=KERNEL_SIZE, strides=1, activation=my_leaky_relu, padding='same', name="d_conv3_2")(x)
    x = layers.Conv2DTranspose(filters=64, kernel_size=(2,2), strides=2, activation=tf.nn.relu, padding='same', name="d_tconv3")(x)
    x = layers.BatchNormalization(name="d_batch_norm3")(x)
    outputs = layers.Activation(tf.keras.activations.sigmoid)(x)

    unet_model = keras.Model(inputs=inputs, outputs=outputs, name="unet")
    unet_model.summary()

    return unet_model

def l1_loss_function(y_actual, y_pred):
    custom_loss = tf.math.abs(tf.math.subtract(y_actual,y_pred))

# def UNet_compile(unet):
#     unet.compile(optimizer='adam',
#                   loss=tf.losses.,
#                   metrics=['accuracy'])
#
#     history = model.fit(train_images, train_labels, epochs=10,
#                         validation_data=(test_images, test_labels))


if __name__ == '__main__':
    unet = UNet_model()
    # UNet_compile(unet)