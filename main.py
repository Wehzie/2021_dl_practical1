import tensorflow as tf
import numpy as np

'''
For an overview over the experimental setup see README.md
'''

# Run experiment 1: MobileNetV2
def run_exp1(x_train, y_train, x_test, y_test):
    
    # Initialize model
    model = tf.keras.applications.MobileNetV2(
    input_shape=(56,56,3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
    )

    # Train and evaluate model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=2)

# Run experiment 2: MobileNet
def run_exp2(x_train, y_train, x_test, y_test):

    # Initialize model
    model = tf.keras.applications.MobileNet(
    input_shape=(56,56,3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
    )

    # Train and evaluate model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=2)

# Run experiment 3: DenseNet121
def run_exp3(x_train, y_train, x_test, y_test):

    # Initialize model
    model = tf.keras.applications.DenseNet121(
    input_shape=(56,56,3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
    )

    # Train and evaluate model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=2)

# Run experiment 4: DenseNet169
def run_exp4(x_train, y_train, x_test, y_test):

    # Initialize model
    model = tf.keras.applications.DenseNet169(
    input_shape=(56,56,3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
    )

    # Train and evaluate model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=2)

# Run experiment 5: 1st of experiments 1 to 4 with MAX pooling
def run_exp5(x_train, y_train, x_test, y_test):
    model = tf.keras.applications.MobileNet(
    input_shape=(56,56,3),
    include_top=False,
    weights="imagenet",
    pooling="max",
    )

    # Train and evaluate model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=1)

# Run experiment 6: 2nd of experiments 1 to 4 with MAX pooling
def run_exp6(x_train, y_train, x_test, y_test):
    model = tf.keras.applications.DenseNet121(
    input_shape=(56,56,3),
    include_top=False,
    weights="imagenet",
    pooling="max",
    )

    # Train and evaluate model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=1)

# Run experiment 7: 1st of experiments 1 to 6 with SDG optimization
def run_exp7(x_train, y_train, x_test, y_test):
    model = tf.keras.applications.DenseNet121(
    input_shape=(56,56,3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
    )

    # Train and evaluate model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='SDG', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=1)

# Run experiment 8: 1st of experiments 1 to 6 with RMSprop optimizer
def run_exp8(x_train, y_train, x_test, y_test):
    model = tf.keras.applications.DenseNet121(
    input_shape=(56,56,3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
    )

    # Train and evaluate model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='RMSprop', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=1)

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
    run_exp5(x_train, y_train, x_test, y_test)
    run_exp6(x_train, y_train, x_test, y_test)

main()