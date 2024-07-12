from random import seed
from random import randrange
import random
import csv

data = []
weights = []

best_perc = None
best_weight = []





# Make a prediction with weights
def classify(row, weights):
    
    total = weights[0]
    for x in range(len(row)-1):
        value = float(row[x])
        weight = float(weights[x+1])
        #print(value, weight)
        product = value * weight
        total += product
    
    if total >= 0:
        return 1 
    else: 
        return 0 

 
#Estimate Perceptron weights using stochastic gradient descent
def train(train_data, n_epoch, l_rate=1):
    for x in range(len(train_data[0])):
        weights.append(float(random.randrange(-100,100)/100))
    #print(weights)
    for y in range(n_epoch):
        total = len(train_data)
        total_right = 0
        for x in range(len(train_data)):
            #print(x)
            predicted = classify(train_data[x], weights)
            actual = train_data[x][-1]
            #print(actual, predicted)
            if actual != predicted:
                weights[0] = weights[0] + (float(actual-predicted))
                for z in range(len(train_data[x])-1):
                    #print( actual, predicted, weights[z+1], z)
                    weights[z+1] = weights[z+1] +(float(actual - predicted) * float(train_data[x][z]))
            else: 
                total_right += 1
        percent_correct = total_right / total
        print(f"epoch {y}.......{percent_correct}% correct")
        if y == 0:
            best_perc = percent_correct
        else:
            if percent_correct > best_perc:
                best_perc = percent_correct
                best_weight = weights

    print(f"best percentage : {best_perc}")




