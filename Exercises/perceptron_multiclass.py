# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
# ----------


# Perceptron implementation
import Helpers.util
from random import randrange
import random


class PerceptronClassifier:

    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.epochs = max_iterations
        self.weights = [] 


    def classify(self, data, weights):
        total = weights[0]
        #print(total)
        for x in range(len(data)-1):
            value = float(data[x])
            weight = float(weights[x+1])
            product = value * weight
            total += product
        return total
 

    def train(self, train_data, labels):
        for y in range(10):
            weight_instance = []
            for x in range(len(train_data[0])):
                weight_instance.append(float(random.randrange(-100,100)/100))
            self.weights.append(weight_instance)
        for y in range(self.epochs):
            total = len(train_data)
            total_right = 0
            for x in range(len(train_data)):
                prediction = []
                for v in range(len(self.weights)):
                    prediction.append(self.classify(train_data[x], self.weights[v]))
                actual = labels[x]
                predicted = max(prediction)
                prediction_index = prediction.index(predicted)
                if actual != prediction_index:
                    self.weights[prediction_index][0] = self.weights[prediction_index][0] + (float(actual-prediction_index))
                    for z in range(len(train_data[x])-1):
                        self.weights[prediction_index][z+1] = self.weights[prediction_index][z+1] -float(train_data[x][z])
                    self.weights[actual][0] = self.weights[actual][0] + (float(actual-prediction_index))
                    for z in range(len(train_data[x])-1):
                        self.weights[actual][z+1] = self.weights[actual][z+1] +float(train_data[x][z])
                else: 
                    total_right += 1
            percent_correct = (total_right / total) * 100
            print(f"epoch {y}.......{percent_correct}% correct")
    

