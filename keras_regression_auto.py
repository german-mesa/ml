import tensorflow as tf
import pandas as pd
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

EPOCHS = 1000
DEBUG_FLAG = True

column_names = [
    'MPG',
    'Cylinders',
    'Displacement',
    'Horsepower',
    'Weight',
    'Acceleration',
    'Model Year',
    'Origin'
]


def load_dataset():
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg"
                                        ".data")

    return pd.read_csv(dataset_path,
                       names=column_names,
                       na_values="?",
                       comment='\t',
                       sep=" ",
                       skipinitialspace=True)


def clean_dataset(dataset):
    # Remove NaN values
    if DEBUG_FLAG:
        print(dataset.isna().sum())

    dataset = dataset.dropna()

    # Encode Origin feature
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

    return dataset


def split_dataset(dataset):
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    return train_dataset, test_dataset


def normalize_dataset(dataset, statistics):
    return (dataset - statistics['mean']) / statistics['std']


def training_statistics(dataset):
    train_stats = dataset.describe()
    return train_stats.transpose()


def inspect_dataset(dataset):
    if DEBUG_FLAG:
        print(dataset.describe())

    sns.pairplot(
        dataset[["MPG", "Cylinders", "Displacement", "Weight"]],
        diag_kind="kde")  # Kernel Density Estimate - [0, 1]

    plt.show()


def inspect_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

    plotter.plot({'Basic': history}, metric="mean_absolute_error")
    plt.ylabel('MAE [MPG]')
    plt.ylim([0, 10])
    plt.show()

    plotter.plot({'Basic': history}, metric="mean_squared_error")
    plt.ylabel('MSE [MPG^2]')
    plt.ylim([0, 20])
    plt.show()


def plot_predictions(labels, predictions):
    plt.axes(aspect='equal')
    plt.scatter(labels, predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.xlim([0, 50])
    plt.ylim([0, 50])
    plt.plot([0, 50], [0, 50])
    plt.show()


def plot_predictions_error(labels, predictions):
    error = predictions - labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    plt.ylabel("Count")
    plt.show()


def main():
    # Load dataset
    dataset = load_dataset()

    # Clean the data
    dataset = clean_dataset(dataset)

    # Split dataset
    train_dataset, test_dataset = split_dataset(dataset)

    # Inspect dataset
    inspect_dataset(train_dataset)

    # Extract labels
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    # Compute training statistics for normalization
    train_stats = training_statistics(train_dataset)

    # Normalize dataset for better convergence
    normed_train_data = normalize_dataset(train_dataset, train_stats)
    normed_test_data = normalize_dataset(test_dataset, train_stats)

    # Build Model
    model = keras.Sequential([
        layers.Dense(64, activation=tf.keras.activations.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.keras.activations.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=[
                      tf.keras.metrics.mean_absolute_error,
                      tf.keras.metrics.mean_squared_error
                  ])

    # Train the model
    history = model.fit(normed_train_data,
                        train_labels,
                        epochs=EPOCHS,
                        validation_split=0.2,
                        verbose=0,
                        callbacks=[
                            # keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                            tfdocs.modeling.EpochDots()
                        ])

    inspect_history(history)

    # Model evaluation
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    # Model predictions
    test_predictions = model.predict(normed_test_data).flatten()

    plot_predictions(test_labels, test_predictions)
    plot_predictions_error(test_labels, test_predictions)


if __name__ == '__main__':
    main()
