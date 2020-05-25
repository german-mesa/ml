import os
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='mean_absolute_error',
        # "no longer improving" being defined as "no better than 1e-3 less"
        min_delta=1e-3,
        # "no longer improving" being further defined as "for at least 5 epochs"
        patience=10,
        verbose=1),
    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(os.getcwd(), 'log'),
        # How often to log histogram visualizations
        histogram_freq=0,
        # How often to log embedding visualizations
        embeddings_freq=0,
        # How often to write logs (default: once per epoch)
        update_freq='epoch'),
]


def function(x):
    noise = np.random.normal(0, 5, x.shape)
    return 2 * np.power(x, 2) + 1 + noise


def main():
    # Load dataset
    x_batch = np.linspace(-10, 10, 500)     # (100, ) Tuple of shape 1
    y_batch = function(x_batch)             # (100, ) Tuple of shape 1

    x_batch = x_batch / max(x_batch)
    y_batch = y_batch / max(y_batch)

    # We need to transform our original data a little bit as we're fitting them to a
    # degree-2 polynomial.
    #
    # The degree-2 polynomial features for [a,b] are:
    #   [1, a, b, a^2, ab, b^2]
    # As we have a 1-D vector [a], the degree-2 polynomial features are:
    #   [1, a, a^2]
    #
    poly = PolynomialFeatures(degree=2)
    x_reshaped = x_batch.reshape(-1, 1)  # (100, 1) Tuple of shape 100, 1
    x_poly = poly.fit_transform(x_reshaped)  # (100, 3) Tuple of shape 100, 3

    # Build model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(units=1, input_shape=[3]),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredError(),
        ]
    )

    # Fit model with training data
    history = model.fit(x_poly,
                        y_batch,
                        callbacks=callbacks,
                        validation_split=0.1,
                        epochs=500)

    # Make predictions using model
    y_prediction = model.predict(x_poly)

    # Plot prediction curve and scattered data
    plt.scatter(x_batch, y_batch)
    plt.plot(x_batch, y_prediction, color='red')
    plt.show()


if __name__ == '__main__':
    main()
