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
model = Sequential()