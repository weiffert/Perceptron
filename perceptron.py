# Author: William Eiffert
# PUID: 0029566085

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import sklearn.datasets

class Perceptron:
    def __init__(self, dimensions, learingRate):
        self.learingRate = learingRate
        self.weights = []
        self.iterations = 0
        for i in range(0, dimensions):
            self.weights.append(0)
        self.bias = 0

    def update(self, point, label):
        if not self.test(point, label):
            self.iterations += 1
            self.bias += self.learingRate * label
            weights = []
            index = 0
            for weight in self.weights:
                weights.append(weight + self.learingRate *
                               label * point[index])
                index += 1
            self.weights = weights
            return True
        return False

    def activation(self, point):
        summation = self.bias
        index = 0
        for weight in self.weights:
            summation += weight * point[index]
            index += 1
        return summation

    def test(self, point, label):
        return label * self.activation(point) > 0

# 2d data


def evaluation(data, learningRate):
    perceptron = Perceptron(len(data.columns) - 1, learningRate)
    change = True
    while change and perceptron.iterations < 1000:
        change = False
        for index, row in data.iterrows():
            if perceptron.update(row, row['label']):
                change = True

    # Graph
    # plot test and train
    red = data[data['label'] == 1]
    blue = data[data['label'] == -1]
    plt.plot(red['x'], red['y'], 'r.')
    plt.plot(blue['x'], blue['y'], 'b.')
    x = np.linspace(data['x'].min() if data['x'].min() > data['y'].min() else data['y'].min(), 
        data['x'].max() if data['x'].max() < data['y'].max() else data['y'].max()
        , 100)
    y = -1 * (perceptron.weights[0] * x + perceptron.bias) / perceptron.weights[1]
    plt.plot(x, y, linestyle='solid')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Perceptron')
    plt.show()
    plt.clf()

def generateData(dimensions):
    separable = False
    #while not separable:
    samples = sklearn.datasets.make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, flip_y=-1)
    red = samples[0][samples[1] == 0]
    blue = samples[0][samples[1] == 1]
    separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])
    points = samples[0].tolist()
    labels = samples[1].tolist()

    index = 0
    data = []
    for x, y in points:
        data.append([x, y, 1 if labels[index] == 1 else -1])
        index += 1
    data = pd.DataFrame(data, columns = ['x', 'y', 'label'])

    return data

# graph perceptron and results.
# ensure linear separability? What about random generations until converges?
# Read in arguments
dimensions = int(sys.argv[1])
learningRate = float(sys.argv[2])
evaluation(generateData(dimensions), learningRate)
