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
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')

# normalization (color range{0-255}
x_train /= 255
x_test /= 255


y_test_label = y_test
# one-hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# build model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

model = Sequential()
# first convolution layer
# make 1 image to 16 images by 5x5 filter weight
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
# pooling layer
# convert 16(28x28 size) images to 16(14x14 size) images.
# the among of images won't be changed, but the images' size change.
model.add(MaxPool2D(pool_size=(2, 2)))
# second convolution layer
# make 16 images to 36 images by 5x5 filter weight
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))
# pooling layer
# convert 36(14x14 size) images to 36(7x7 size) images.
# the among of images won't be changed, but the images' size change.
model.add((MaxPool2D(pool_size=(2, 2))))
# drop some neurons every round to avoid overfitting
model.add(Dropout(0.25))
# flatten layer
model.add(Flatten())
# hidden layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# output layer
model.add(Dense(10, activation='softmax'))

# check model details
print(model.summary())

# define how to train the model
# evaluate by accuracy, set up loss function and then go back to fix the model
# there are more than 2 results, so use categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model and store the history
# using 20% of training data as validation data to evaluate the accuracy
# separate the rest of training data by batch_size and train the model 10 rounds.
train_history = model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=10, batch_size=300, verbose=2)

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

'''
The result shows 0.9914 accuracy.
Check out the plots on accuracy and loss, there's no overfitting problem.
The accuracy and loss of train and validation are closing to the same amount.
'''
