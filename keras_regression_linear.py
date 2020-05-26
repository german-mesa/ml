import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Function is y = 3x + 1
def linear_function(x):
    noise = np.random.normal(0, 1, x.shape)
    return 3 * x + 1 + noise


checkpoint_path = os.path.join(os.getcwd(), 'checkpoint/') + 'cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-3 less"
        min_delta=1e-6,
        # "no longer improving" being further defined as "for at least 5 epochs"
        patience=10,
        verbose=1),
    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(os.getcwd(), 'log'),
        # log_dir=os.path.join(os.getcwd(), 'log/{}'.format(time())),
        # How often to log histogram visualizations
        histogram_freq=0,
        # How often to log embedding visualizations
        embeddings_freq=0,
        # How often to write logs (default: once per epoch)
        update_freq='epoch'),
]


def main():
    # Load training data from function
    x_batch = np.linspace(start=-10.0, stop=10.0, num=100)
    y_batch = np.array(linear_function(x_batch), dtype=float)

    # Build model
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(units=1, input_shape=[1])]
    )

    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[
                      tf.keras.metrics.MeanAbsoluteError(),
                      tf.keras.metrics.MeanSquaredError(),
                  ]
                  )

    # Fit model with training data
    history = model.fit(x=x_batch,
                        y=y_batch,
                        callbacks=callbacks,
                        validation_split=0.1,
                        epochs=50)

    # Make predictions using model
    y_pred_batch = model.predict(x_batch)

    # Plot scattered data and function fit
    plt.scatter(x_batch, y_batch)
    plt.plot(x_batch, y_pred_batch, color='red')
    plt.title('Function')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()


if __name__ == '__main__':
    main()
