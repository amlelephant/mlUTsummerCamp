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


# Perceptron implementation for imitation learning
import Helpers.util
from Exercises.perceptron_multiclass import PerceptronClassifier
from random import randrange
from pacman import GameState
import random


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.epochs = maxIterations
        self.weights = [] 
        

    def convert_data(self, data):
        # fix datatype issues
        # if it comes in inside of a list, pull it out of the list
        if isinstance(data, list):
            data = data[0]

        #data comes as a tuple
        all_moves_features = data[0] #grab the features (dict of action->features)
        legal_moves = data[1]        #grab the list of legal moves from this state
        return_features = {}
        #loop each action
        for key in all_moves_features:
            #convert feature values from dict to list
            list_features = list(all_moves_features[key].values()) 
            return_features[key] = list_features

        return (return_features, legal_moves) 


    def classify(self, data):
        #leave this call to convert_data here!
        features, legal_moves = self.convert_data(data)

        ##your code goes here##
        results = []
        
        for y in legal_moves:
            total = self.weights[0]
            for x in range(len(features[y])-1):
                value = float(features[y][x])
                weight = float(self.weights[x+1])
                product = value * weight
                total += product
            results.append(total)
 
        #your predicted_label needs to be returned inside of a list for the PacMan game
        predicted_label = legal_moves[results.index(max(results))]
        return [predicted_label] 


    def train(self, train_data, labels):
        features, legal_moves = self.convert_data(train_data[0])
        firstElement = features['East']
        for x in range(len(firstElement)):
            self.weights.append(float(random.randrange(-100,100)/100))
        
        for y in range(self.epochs):
            total = len(train_data)
            total_right = 0
            for x in range(len(train_data)):
                predicted = self.classify(train_data[x])
                features, legal_moves = self.convert_data(train_data[x])
                actual = labels[x]
                if actual != predicted[0]:
                    self.weights[0] = self.weights[0] + features[actual][0]
                    for z in range(len(train_data[x])-1):
                        self.weights[z+1] = self.weights[z+1] + features[actual][z]
                    self.weights[0] = self.weights[0] - features[predicted[0]][0]
                    for z in range(len(train_data[x])-1):
                        self.weights[z+1] = self.weights[z+1] - features[predicted[0]][z]
                else: 
                    total_right += 1
            percent_correct = total_right / total
            print(f"epoch {y}.......{percent_correct}% correct")