import numpy as np
import math

# MNIST STUFF - COMMENT OUT IF NOT USING
import v2_imageParser as ip 
from matplotlib import pyplot as plt 
TESTING_DATA, TESTING_LABELS = ip.get_training() 
TRAINING_DATA, TRAINING_LABELS = ip.get_testing() 
# MNIST STUFF - COMMENT OUT IF NOT USING

# OTHER - COMMENT IN IF NOT USING MNIST
# TESTING_DATA, TESTING_LABELS = 
# TRAINING_DATA, TRAINING_LABELS = 
# OTHER - COMMENT IN IF NOT USING MNIST

# Configuration settings
MODE = "TRAIN" # TRAIN, TEST or OTHER

LOAD_FILE = True
SAVE_FILE_AFTER_TRAINING = True
FILE_PATH = "./weights.npy"

TRAINING_ITERATIONS = 1000
LEARNING_RATE = 0.01
LAYERS=[784, 6, 6, 6, 10]


# This is the only function that needs code to be changed for MNIST
def make_prediction(index, Weights, Biases):
    img = TESTING_DATA[:, index].reshape(28, 28) * 255

    predictions, _ = forward_prop(TESTING_DATA[:, index, None], Weights, Biases) # _ is the Z values, we don't need them - we just want the predictions
    prediction = get_predictions(predictions[-1]) # the predictions are for each layer, so just select the last one as that is the output layer
    
    # MNIST STUFF - COMMENT OUT IF NOT USING MNIST
    plt.xlabel(f"Prediction: {prediction}, Actual: {TESTING_LABELS[index]}")
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.show()
    # MNIST STUFF - COMMENT OUT IF NOT USING MNIST




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

def forward_prop(Data, Weights, Biases):
    # Raw is used to store raw values for backpropagation later
    # Filtered is used to store the values after the activation function, this is the output of the layer
    Raw = []
    Filtered = []
    last = Data

    for i in range(len(Weights)):
        # print(f"\n\nWeights {i} shape: {Weights[i].shape}\nBiases {i} shape: {Biases[i].shape}\n last shape: {last.shape}")
        # print(f"Max value in last: {np.max(last)}")
        Raw.append(Weights[i].dot(last) + Biases[i])
        if (i == len(Weights) - 1):
            # print(f"Layer: {i}, using softmax")
            # print(Z[i].shape)
            Filtered.append(softmax(Raw[i]))
        else:
            # print(f"Layer: {i}, using ReLU")
            Filtered.append(ReLU(Raw[i]))

        last = Filtered[i]

    return Filtered, Raw

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # +1 to account for 0 index
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T # transpose to make it a column vector
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

def gradient_descent(X, Y, iterations, alpha, layers=[784, 6, 6, 6, 10], Weights=None, Biases=None):
    # Train the network using gradient descent
    if Weights is None or Biases is None:
        Weights, Biases = init_params(layers)

    for i in range(iterations):
        A, Z = forward_prop(X, Weights, Biases)
        dW, dB = back_prop(A, Z, Weights, X, Y)
        Weights, Biases = update_params(dW, dB, Weights, Biases, alpha)

        if (i % math.ceil(iterations / 25) == 0): # print every 1/100th of the iterations
            predictions = get_predictions(A[-1])
            print(f"\nIteration: {i}\nAccuracy: {get_accuracy(predictions, Y)*100}%")

    print(f"Final Accuracy: {get_accuracy(get_predictions(A[-1]), Y)*100}%")
    return Weights, Biases

# HANDLING LOADING/SAVING WEIGHTS AND BIASES
def load_network():
    print("Loading network...")
    return np.load(FILE_PATH, allow_pickle=True)

def save_network(data):
    print("Saving network...")
    np.save(FILE_PATH, data)

def train():
    Weights, Biases = [None]*2

    if LOAD_FILE:
        data = load_network()
        Weights, Biases = data[0], data[1]
    
    Weights, Biases = gradient_descent(TRAINING_DATA, TRAINING_LABELS, TRAINING_ITERATIONS, LEARNING_RATE, layers=LAYERS, Weights=Weights, Biases=Biases)

    if SAVE_FILE_AFTER_TRAINING:
        data = [Weights, Biases]
        save_network(data)

    print("Finished training")
    return Weights, Biases

def process_input(input, Weights=None, Biases=None):
    if Weights is None or Biases is None:
        Weights, Biases = load_network()
    
    predictions, _ = forward_prop(input, Weights, Biases) # _ is the Z values, we don't need them - we just want the predictions
    return predictions # the predictions are for each layer, bare in mind for outputting


def run():
    if MODE == "TRAIN":
        train()
    elif MODE == "TEST":
        Weights, Biases = load_network()
        make_prediction(0, Weights, Biases)
    else:
        print(f"Network launched in mode {MODE}, waiting.")


run()