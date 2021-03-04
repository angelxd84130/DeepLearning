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

    # load positive texts
    for file in os.listdir(positive_path):
        text_path = positive_path + file
        with open(text_path, 'r', encoding='utf8') as text:
            texts += [text.readline()]
    pos_length = len(texts)

    # load negative texts
    for file in os.listdir(negative_path):
        text_path = negative_path + file
        with open(text_path, 'r', encoding='utf8') as text:
            texts += [text.readline()]
    neg_length = len(texts)-pos_length
    labels = [1]*pos_length + [0]*neg_length
    print("get positive data:", pos_length, "negative data:", neg_length)
    return (texts, labels)

(x_train, y_train) = read_files("train")
(x_test, y_test) = read_files("test")

'''
# get datasets from keras as a backup plan
from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data()
'''