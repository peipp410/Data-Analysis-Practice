from tensorflow.keras.datasets.mnist import load_data
from numpy import unique
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from keras.utils import to_categorical
import random

# preprocess
(x_train, y_train), (x_test, y_test) = load_data()
random.seed(1)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
in_shape = x_train.shape[1:]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
n_classes = len(unique(y_train))


# define model
model = Sequential()


# Convolution layer with 32 4 by 4 filters, the activation is relu
model.add(Conv2D(filters=32, kernel_size=(4, 4), activation='relu', input_shape=in_shape))
# Max pooling layer with 2 by 2 pooling window.
model.add(MaxPool2D(pool_size=(2, 2)))
# Convolution layer with 64 4 by 4 filters, the activation is relu
model.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))
# Max pooling layer with 2 by 2 pooling window.
model.add(MaxPool2D(pool_size=(2, 2)))
# Drop 10% of the data
model.add(Dropout(0.1))
# Flatten layer
model.add(Flatten())

# # First hidden layer with 800 hidden nodes
model.add(Dense(units=800, activation='relu'))
# The output layer with 10 classes output.
# Use the softmax activation function for classification
model.add(Dense(units=10, activation='softmax'))

# define loss function and optimizer
# set the optimizer to 'sgd', then you may switch to 'adam'.
# use cross entropy as the loss for multi-class classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=2, validation_data=(x_test, y_test))

# evaluate the model
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test Accuracy on the test set: %.3f' % acc)

model.summary()
