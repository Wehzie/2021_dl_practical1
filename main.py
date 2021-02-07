import tensorflow as tf
import numpy as np

def main():
    mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print("x_train shape:", x_train.shape)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print("x_train shape:", x_train.shape)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)

    size = (224,224)
    x_train = x_train.map(lambda image, label: (tf.image.resize(image, size), label))
    print("x_train shape:", x_train.shape)

    # pretrained model
    model = tf.keras.applications.MobileNetV2(
    #input_shape=Input(shape(32,28,28)),
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax"
    )

    x_train = tf.keras.applications.mobilenet_v2.preprocess_input(x_train)
    y_train = tf.keras.applications.mobilenet_v2.preprocess_input(y_train)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=2)

main()