from keras.utils import np_utils
import numpy as np
np.random.seed(10)

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# If having problems on loading, try to install python certificates
# or use the following code instead.

'''
with np.load("mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
'''

x_test_image = x_test

# convert it from 2D to 1D
# original shape(60000, 28, 28)
x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')

# normalization (color range{0-255}
x_train /= 255
x_test /= 255


y_test_label = y_test
# one-hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# build model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
model = Sequential()
# input layer -> hidden layer
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))
'''
# use more neurons in a layer to increase accuracy
model.add(Dense(units=256, input_dim=1000, kernel_initializer='normal', activation='relu'))
# drop some neurons every round to avoid overfitting
model.add(Dropout(0.5))
'''

# hidden layer -> output layer
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# check model details
print(model.summary())

# define how to train the model
# evaluate by accuracy, set up loss function and then go back to fix the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model and store the history
# using 20% of training data as validation data to evaluate the accuracy
# separate the rest of training data by batch_size and train the model 10 rounds.
train_history = model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=10, batch_size=200, verbose=2)

# show train history
import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.ylabel(train)
    plt.legend(['train', 'validation'])
    plt.show()


# using accuracy and loss plots helps to check overfitting problem
print(show_train_history(train_history, 'accuracy', 'val_accuracy'))
print(show_train_history(train_history, 'loss', 'val_loss'))

# evaluation
scores = model.evaluate(x_test, y_test)
print('Final Accuracy:', scores[1])

# show prediction result
# use predict_classes to convert one-hot encoding back to digits
prediction = model.predict_classes(x_test)
print(prediction)

# plot results(real/predict values)
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    for i in range(0, num):
        ax = plt.subplot(5, 5, i+1)
        ax.imshow(images[idx], cmap='binary')
        title = 'label=' + str(labels[idx])
        if len(prediction) > 0:
            title += ',predict=' + str(prediction[idx])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx=340)

# confusion matrix
import pandas as pd
print(pd.crosstab(y_test_label, prediction, colnames=['predict'], rownames=['label']))

