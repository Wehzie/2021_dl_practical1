import tensorflow as tf
import numpy as np

def rescale(a):
    return np.repeat(a[..., np.newaxis], 3, -1)

def main():
    # Import data
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Upscale 28x28 by a factor of 2 to 56x56
    x_train = np.kron(x_train, np.ones((2,2)))
    x_test = np.kron(x_test, np.ones((2,2)))

    # Rescale grayscale images to "RGB"-images
    x_train = rescale(x_train)
    x_test = rescale(x_test)
    
    # Pretrained model
    model = tf.keras.applications.MobileNetV2(
    input_shape=(56,56,3),
    alpha=1.0,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling="avg",
    classes=1000,
    classifier_activation="softmax"
    )

    # Preprocess training input
    #x_train = tf.keras.applications.mobilenet_v2.preprocess_input(x_train)
    #y_train = tf.keras.applications.mobilenet_v2.preprocess_input(y_train)

    # Train & evaluate model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=2)

main()