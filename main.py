import tensorflow as tf
import numpy as np

def main():
    # Import data
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Upscale 28x28 by a factor of 2 to 56x56
    x_train = np.kron(x_train, np.ones((2,2)))
    x_test = np.kron(x_test, np.ones((2,2)))

    # Rescale grayscale images to "RGB"-images
    x_train = np.repeat(x_train[..., np.newaxis], 3, -1)
    x_test = np.repeat(x_test[..., np.newaxis], 3, -1)
    
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

    # Train & evaluate model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=2)

main()