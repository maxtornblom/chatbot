# Importing required libraries
import nltk
nltk.download("punkt")
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

# Checking the version of tensorflow
print(tensorflow.__version__)

# Loading data from json 
with open("intents.json") as file:
    data = json.load(file)

# Loading data from pickle file
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

# Creating a list of words, labels, and documents
words = []
labels = []
docs_x = []
docs_y = []

# Tokenizing the patterns, adding them to the document list, and extending the word list
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    # Adding the tag to the labels list if it is not already present
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Stemming and sorting the word list
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

# Sorting the label list
labels = sorted(labels)

# Creating training and output lists with bags of words for each document
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

# Converting training and output lists into numpy arrays
training = numpy.array(training)
output = numpy.array(output)

# Saving the data into a pickle file
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

# Defining the neural network architecture using tflearn
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Initializing the model
model = tflearn.DNN(net, tensorboard_dir="tflearn_logs")

# Training the model on the data
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)

# Saving the model
model.save("model.tflearn")

# Printing a message when training is complete
print("training complete")
