import tensorflow as tf
from tensorflow import keras


class MNISTClassifierModel(tf.keras.Sequential):
    def __init__(self, name=None):
        super(MNISTClassifierModel, self).__init__(name=name)

        self.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
        self.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255))
        self.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    def compile(self):
        super(MNISTClassifierModel, self).compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                                                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                                  metrics=['accuracy'])


def classifier_model(dataset):
    # Load dataset
    (images_train, labels_train), (images_test, labels_test) = dataset.load_data()

    print('Images loaded from Keras:')
    print(f'#Training Images: {len(labels_train)}')
    print(f'#Test Images: {len(labels_test)}')

    # Rescaling values to [0, 1]
    # images_train = images_train / 255.
    # images_test = images_test / 255.

    # Define model using our class above
    print('Define model with one layer of 128 neurons and 10 different categories')
    model = MNISTClassifierModel()

    # Configure the model for training
    print('Configure the model for training')
    model.compile()

    # Fit the model to the training dataset
    print('Fit the model to the training dataset')
    model.fit(images_train, labels_train, epochs=10)

    model.summary()

    # Evaluate model loss and accuracy
    print('Evaluate model loss and accuracy')
    test_loss, test_acc = model.evaluate(images_test, labels_test)

    print(f'Loss: {test_loss}')
    print(f'Accuracy: {test_acc}')


if __name__ == '__main__':
    # Digits classification
    print('Digits classification')
    classifier_model(keras.datasets.mnist)

    # Fashion Dataset
    print('Fashion Dataset')
    classifier_model(keras.datasets.fashion_mnist)


# https://keras.io/examples/vision/image_classification_from_scratch/




