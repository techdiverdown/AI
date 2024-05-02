import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version is: ", tf.__version__)

# Load the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the images to [0, 1] range
# This is done by dividing by 255.0 since the values are RBG values 0-255.
# 0 being white and 255 being black
train_images = train_images / 255.0
test_images = test_images / 255.0

# LIst of class names that correspond to the labels in the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Building the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the image from 28x28 pixels to a 1D array of 784 pixels
    tf.keras.layers.Dense(256, activation='relu'),  # First layer with 256 neurons, using ReLU activation function
    tf.keras.layers.Dropout(0.2),  # Dropout layer to reduce overfitting by ignoring randomly selected neurons during training
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 neurons for each class, using softmax to output probabilities
])

# Compiling the model
# 'adam' optimizer is used for improving upon the classical stochastic gradient descent
# 'sparse_categorical_crossentropy' is used as the loss function for multi-class classification problems
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Ensure loss is spelled correctly
              metrics=['accuracy'])

# Early stopping to prevent overfitting by stopping the training when the validation loss is not improving
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Training the model
# Here, we train the model for up to 30 epochs (or passes through the dataset)
# We also use 10% of the training data for validation to monitor accuracy during training
model.fit(train_images, train_labels, epochs=30, validation_split=0.1, callbacks=[early_stopping])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# lets make a prediction
predictions = model.predict(test_images)
# take the first prediction and print the corresponding class name
print(class_names[np.argmax(predictions[0])])

