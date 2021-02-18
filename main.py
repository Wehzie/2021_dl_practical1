import tensorflow as tf
import numpy as np

'''
For an overview over the experimental setup see README.md
'''

# run experiment 1: MobileNetV2
def run_exp1(x_train, y_train, x_test, y_test):
    
    # initialize model
    model = tf.keras.applications.MobileNetV2(
    input_shape=(56,56,3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
    )

    # train and evaluate model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=2)

# run experiment 2: NasNetMobile
def run_exp2(x_train, y_train, x_test, y_test):

    # initialize model
    model = tf.keras.applications.NASNetMobile(
    input_shape=(56,56,3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
    )

    # train and evaluate model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=2)

# run experiment 3: DenseNet121
def run_exp3(x_train, y_train, x_test, y_test):

    # initialize model
    model = tf.keras.applications.DenseNet121(
    input_shape=(56,56,3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
    )

    # train and evaluate model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=2)

# run experiment 4: Xception
def run_exp4(x_train, y_train, x_test, y_test):

    # initialize model
    model = tf.keras.applications.Xception(
    input_shape=(56,56,3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
    )

    # train and evaluate model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=2)

# experiment 5: 1st of experiments 1 to 4 with MAX pooling
def run_exp5():
    pass

# experiment 6: 2nd of experiments 1 to 4 with MAX pooling
def run_exp6():
    pass

# experiment 7: 1st of experiments 1 to 6 with SDG optimization
def run_exp7():
    pass

# experiment 8: 1st of experiments 1 to 6 with RMSprop optimizer
def run_exp8():
    pass

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

    # Run experiments
    run_exp2(x_train, y_train, x_test, y_test)

main()