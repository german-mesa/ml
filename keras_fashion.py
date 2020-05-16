import tensorflow as tf
from tensorflow import keras


# Load MNIST Fashion Dataset
fashion_mnist = keras.datasets.fashion_mnist
(images_train, labels_train), (images_test, labels_test) = fashion_mnist.load_data()

print('Images loaded from Keras:')
print(f'#Training Images: {len(labels_train)}')
print(f'#Test Images: {len(labels_test)}')

images_train = images_train / 250.0
images_test = images_test / 250.0

# Define model with one layer of 128 neurons and 10 different categories
print('Define model with one layer of 128 neurons and 10 different categories')
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ]
)


# Configure the model for training
print('Configure the model for training')
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


# Fit the model to the training dataset
print('Fit the model to the training dataset')
model.fit(images_train, labels_train, epochs=10)

model.summary()

# Evaluate model loss and accuracy
print('Evaluate model loss and accuracy')
test_loss, test_acc = model.evaluate(images_test, labels_test)

print(f'Loss: {test_loss}')
print(f'Accuracy: {test_acc}')

