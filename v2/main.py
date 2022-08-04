import numpy as np
from matplotlib import pyplot as plt
import v2_imageParser as ip
import math

X_TESTING, Y_TESTING = ip.get_training()
X_TRAINING, Y_TRAINING = ip.get_testing()

# currenctly called as init_params([784, 6, 6, 6, 10])
def init_params(layers: list):
    Weights = [np.random.randn(layers[i+1], layers[i]) for i in range(len(layers)-1)]
    Biases = [np.random.randn(layers[i+1], 1) for i in range(len(layers)-1)]

    return Weights, Biases

def ReLU(inp):
    return np.maximum(0, inp)

def scale(inp):
    return inp / np.max(inp)

def softmax(inp):
    # print(f"Max Exp: {np.max(inp)}")
    return np.exp(inp) / sum(np.exp(inp))

def forward_prop(X, Weights, Biases):
    Z = []
    A = []
    last = X

    for i in range(len(Weights)):
        # print(f"\n\nWeights {i} shape: {Weights[i].shape}\nBiases {i} shape: {Biases[i].shape}\n last shape: {last.shape}")
        # print(f"Max value in last: {np.max(last)}")
        Z.append(Weights[i].dot(last) + Biases[i])
        if (i == len(Weights) - 1):
            # print(f"Layer: {i}, using softmax")
            # print(Z[i].shape)
            A.append(softmax(Z[i]))
        else:
            # print(f"Layer: {i}, using ReLU")
            A.append(ReLU(Z[i]))

        last = A[i]

    return A, Z

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(inp):
    return inp > 0

def back_prop(A, Z, Weights, X, Y):
    dZ = [0] * len(Weights) # derivative of Z
    dW = [0] * len(Weights) # derivetive of weights
    dB = [0] * len(Weights) # derivative of biases
    one_hot_Y = one_hot(Y)
    m = Y.size

    # last layer
    dZ[-1] = A[-1] - one_hot_Y # derivative of Z for the last layer
    dW[-1] = 1 / m * dZ[-1].dot(A[-2].T) # derivative of W for the last layer
    dB[-1] = 1 / m * np.sum(dZ[-1]) # derivative of B for the last layer
    # print(f"Max Bias Difference: {np.max(np.abs(dB[-1]))}")

    # other layers
    for i in range(len(Weights)-2, -1, -1): # from the second last layer to the first, looping backwards
        dZ[i] = Weights[i+1].T.dot(dZ[i+1]) * deriv_ReLU(Z[i]) # derivative of Z for the current layer

        # Define whether to use the pervious layer's calculated weights or the input layer's weights
        tmp = None
        if i == 0: # if the layer previous to the current layer is the input layer
            tmp = X     # input layer
        else:
            tmp = A[i-1] # previous layer

        dW[i] = 1 / m * dZ[i].dot(tmp.T) # derivative of W for the current layer
        dB[i] = 1 / m * np.sum(dZ[i]) # derivative of B for the current layer

    return dW, dB

def update_params(dW, dB, Weights, Biases, alpha):
    for i in range(len(Weights)):
        # print(f"\n\nRunning for layer: {i}\nWeights {i} shape: {Weights[i].shape}\nBiases {i} shape: {Biases[i].shape}\n dW {i} shape: {dW[i].shape}\n dB {i}: {dB[i]}")
        Weights[i] -= alpha * dW[i]
        Biases[i] -= alpha * dB[i]

    return Weights, Biases

def get_predictions(A):
    return np.argmax(A, 0)

def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    Weights, Biases = init_params([784, 6, 6, 10])

    for i in range(iterations):
        A, Z = forward_prop(X, Weights, Biases)
        dW, dB = back_prop(A, Z, Weights, X, Y)
        Weights, Biases = update_params(dW, dB, Weights, Biases, alpha)

        if (i % math.ceil(iterations / 25) == 0): # print every 1/100th of the iterations
            predictions = get_predictions(A[-1])
            print(f"\nIteration: {i}\nAccuracy: {get_accuracy(predictions, Y)*100}%")

    print(f"Final Accuracy: {get_accuracy(get_predictions(A[-1]), Y)*100}%")
    return Weights, Biases


Weights, Biases = gradient_descent(X_TRAINING, Y_TRAINING, 50000, 0.1)