import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-5 less"
        min_delta=1e-5,
        # "no longer improving" being further defined as "for at least 5 epochs"
        patience=5,
        verbose=1),
    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(os.getcwd(), 'log'),
        # How often to log histogram visualizations
        histogram_freq=0,
        # How often to log embedding visualizations
        embeddings_freq=0,
        # How often to write logs (default: once per epoch)
        update_freq='epoch')
]


# Function is y = 3x + 1
def linear_function(x):
    return 3 * x + 1


def build_model():
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(units=1, input_shape=[1])]
    )

    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[
                      tf.keras.metrics.MeanAbsoluteError(),
                      tf.keras.metrics.MeanSquaredError(),
                  ], )
    return model


def plot_history(history):
    plt.figure()

    plt.ylim([0, 0.5])

    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error / Mean Squared Error')
    plt.legend()

    plt.plot(history.epoch,
             np.array(history.history['mean_absolute_error']),
             label='Mean Absolute Error')
    plt.plot(history.epoch,
             np.array(history.history['mean_squared_error']),
             label='Mean Squared Error')

    plt.show()


def main():
    xs = np.linspace(start=-10.0, stop=10.0, num=1000)
    ys = np.array(linear_function(xs), dtype=float)

    model = build_model()
    history = model.fit(x=xs,
                        y=ys,
                        callbacks=callbacks,
                        validation_split=0.1,
                        epochs=50)

    # plot_history(history)

    print(model.predict([10.0, 5.0]))


if __name__ == '__main__':
    main()
