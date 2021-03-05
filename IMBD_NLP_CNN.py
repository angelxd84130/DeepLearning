import urllib.request
import os
'''
# download dataset by url
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
# set up the destination for the file
filepath = "aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print('downloaded:', result)

# unzip file
import tarfile
if not os.path.exists("aclImdb"):
    tfile = tarfile.open("aclImdb_v1.tar.gz", 'r:gz')
    result = tfile.extractall('')
'''
# read files
def read_files(filetype):
    path = 'aclImdb/'
    positive_path = path + filetype + '/pos/'
    negative_path = path + filetype + '/neg/'
    texts = []

    # remove html tags
    import re
    re_tag = re.compile(r'<[^>]+>')

    # load positive texts
    for file in os.listdir(positive_path):
        text_path = positive_path + file
        with open(text_path, 'r', encoding='utf8') as text:
            texts += [re_tag.sub('', text.readline())]
    pos_length = len(texts)

    # load negative texts
    for file in os.listdir(negative_path):
        text_path = negative_path + file
        with open(text_path, 'r', encoding='utf8') as text:
            texts += [re_tag.sub('', text.readline())]
    neg_length = len(texts)-pos_length
    labels = [1]*pos_length + [0]*neg_length
    print("get positive data:", pos_length, "negative data:", neg_length, "from", filetype)
    return (texts, labels)


(x_train, y_train) = read_files("train")
(x_test, y_test) = read_files("test")

'''
# get datasets from keras as a backup plan
from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data()
'''
#print(x_train[:2])
#print(x_test[:2])

# set up a dictionary with 2000 high-frequency words from texts
from keras.preprocessing.text import Tokenizer
token = Tokenizer(num_words=2000)
token.fit_on_texts(x_train)
# check out the dictionary
print(token.word_index)

# convert words to digits from the texts
x_train = token.texts_to_sequences(x_train)
x_test = token.texts_to_sequences(x_test)
# limit the length of every text
from keras.preprocessing import sequence
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)

# set up training model
from keras import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
model = Sequential()
# Embedding layer
# input 2000 training examples, and each length is 100
model.add(Embedding(output_dim=32, input_dim=2000, input_length=100))
model.add(Dropout(0.25))
# Flatten layer
# make 32x100=3200 neurons
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=1, activation='sigmoid'))
print(model.summary())
# there are only 2 results, so use categorical_crossentropy
model.compile(loss='binary_crossentropy', optimizer='adm', metrics=['accuracy'])

# start training
# train an example pre time (text length=100), and train all example 10 times
train_history = model.fit(x_train, y_train, batch_size=100, epochs=10, verbose=2, validation_split=0.2)

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
print("Final accuracy", scores[1])

prediction = model.predict_classes(x_test)
# show real value and prediction value

