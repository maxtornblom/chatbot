# Importing necessary libraries
import pickle
import tflearn
import numpy
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os

# Loading the data from the 'data.pickle' file
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

# Loading the intents from the 'intents.json' file
with open("intents.json") as file:
    data = json.load(file)

# Initializing the stemmer
stemmer = LancasterStemmer()

# Building the neural network model
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Loading the pre-trained model from the 'model.tflearn' file
model.load("model.tflearn")

# Function to convert the user's input to a bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

# Function to handle the user's conversation with the chatbot
def chat():
    print("start talking to bot!")
    responses = []
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Predicting the output from the user's input using the pre-trained model
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        accuracy = results[results_index]

        # Getting the appropriate response from the 'intents.json' file based on the predicted tag
        if results[results_index] > 0:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
                    break

            # Printing the response and the accuracy of the predicted tag
            print("bot:", responses)
            print(accuracy * 100, '%')
        else:
            # The bot asks for the user to rephrase the input
            print("Bot: didn't get that, try again")

# Clearing the console and starting the chatbot
os.system('cls')
chat()


