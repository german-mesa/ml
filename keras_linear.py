import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time


# Function is y = 3x + 1
def linear_function(x):
    return 3 * x + 1


checkpoint_path = os.path.join(os.getcwd(), 'checkpoint/') + 'cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-3 less"
        min_delta=1e-3,
        # "no longer improving" being further defined as "for at least 5 epochs"
        patience=5,
        verbose=1),
    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(os.getcwd(), 'log/{}'.format(time())),
        # How often to log histogram visualizations
        histogram_freq=0,
        # How often to log embedding visualizations
        embeddings_freq=0,
        # How often to write logs (default: once per epoch)
        update_freq='epoch'),
    tf.keras.callbacks.ModelCheckpoint(
        # Create a callback that saves the model's weights
        filepath=checkpoint_path,
        save_weights_only=True,
        # Save the model every 5 epochs
        period=5,
        verbose=1)
]


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


def main():
    xs = np.linspace(start=-10.0, stop=10.0, num=1000)
    ys = np.array(linear_function(xs), dtype=float)

    model = build_model()
    history = model.fit(x=xs,
                        y=ys,
                        callbacks=callbacks,
                        validation_split=0.1,
                        epochs=50)

    print(model.predict([10.0, 5.0]))


if __name__ == '__main__':
    main()
