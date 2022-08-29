import numpy as np
import random
import math

# MNIST STUFF - COMMENT OUT IF NOT USING
import imageParser as ip
from matplotlib import pyplot as plt

TESTING_DATA, TESTING_LABELS = ip.get_training()
TRAINING_DATA, TRAINING_LABELS = ip.get_testing()
# MNIST STUFF - COMMENT OUT IF NOT USING

# OTHER - COMMENT IN IF NOT USING MNIST
# TESTING_DATA, TESTING_LABELS = 
# TRAINING_DATA, TRAINING_LABELS = 
# OTHER - COMMENT IN IF NOT USING MNIST


# Configuration settings
MODE = "TEST"  # TRAIN, TEST or OTHER

LOAD_FILE = True
SAVE_FILE_AFTER_TRAINING = True
FILE_PATH = "../weights.npy"

TRAINING_ITERATIONS = 1000  # How many training iterations to run
LEARNING_RATE = 0.05  # 0.5 is the default for a new network
LEARNING_DECAY = 0.999  # This is multiplied by the current learning rate after each iteration to decay it
LAYERS = [784, 100, 20, 6, 6, 6,
          10]  # The number of nodes/neurons in each layer: [inputs, ...h, outputs] where h is the hidden layers

# End of Configuration Settings

# GLOBALS
global Learning_Current
Learning_Current = LEARNING_RATE


# This is the only function that needs code to be changed for MNIST
def make_prediction(weights, biases, index=None, display=False):
    if index is None:
        index = random.randint(0, len(TESTING_DATA[:]))

    input_data = TESTING_DATA[:, index, None]

    calculated_nodes, _ = forward_prop(input_data, weights, biases)
    # _ is the Z values, we don't need them - we just want the predictions

    prediction = get_predictions(calculated_nodes[-1])[0]
    # the nodes calculated are for every layer, so just select the last one as that is the output layer

    output_string = f"Prediction: {prediction}, Actual: {TESTING_LABELS[index]}"

    if display:
        # MNIST STUFF - COMMENT OUT IF NOT USING MNIST OR KEEP DISPLAY=FALSE
        plt.xlabel(output_string)
        img = TESTING_DATA[:, index].reshape(28, 28) * 255
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.show()
        # MNIST STUFF - COMMENT OUT IF NOT USING MNIST OR KEEP DISPLAY=FALSE
    else:
        print(output_string)

    return {"inputs": input_data, "prediction": prediction, "all_nodes": calculated_nodes, "actual": TESTING_LABELS[index]}


def init_params(layers: list):
    Weights = [np.random.rand(layers[i + 1], layers[i]) - 0.5 for i in range(len(layers) - 1)]
    Biases = [np.random.rand(layers[i + 1], 1) - 0.5 for i in range(len(layers) - 1)]

    return Weights, Biases


def ReLU(inp):
    return np.maximum(0, inp)


def scale(inp):
    return inp / np.max(inp)


def softmax(inp):
    # print(f"Max Exp: {np.max(inp)}")
    return np.exp(inp) / sum(np.exp(inp))


def forward_prop(data, weights, biases):
    # Raw is used to store raw values for backpropagation later
    # Filtered is used to store the values after the activation function, this is the output of the layer
    Raw = []
    Filtered = []
    last = data

    for i in range(len(weights)):
        # print(f"\n\nWeights {i} shape: {weights[i].shape}\nBiases {i} shape: {biases[i].shape}\n last shape: "
        #       f"{last.shape}")
        # print(f"Max value in last: {np.max(last)}")
        Raw.append(weights[i].dot(last) + biases[i])
        if i == len(weights) - 1:
            # print(f"Layer: {i}, using softmax")
            # print(Z[i].shape)
            Filtered.append(softmax(Raw[i]))
        else:
            # print(f"Layer: {i}, using ReLU")
            Filtered.append(ReLU(Raw[i]))

        last = Filtered[i]

    return Filtered, Raw


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))  # +1 to account for 0 index
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T  # transpose to make it a column vector
    return one_hot_Y


def derive_ReLU(inp):
    return inp > 0


def back_prop(A, Z, Weights, X, Y):
    dZ = [0] * len(Weights)  # derivative of Z
    dW = [0] * len(Weights)  # derivative of weights
    dB = [0] * len(Weights)  # derivative of biases
    one_hot_Y = one_hot(Y)
    m = Y.size

    # last layer
    dZ[-1] = A[-1] - one_hot_Y  # derivative of Z for the last layer
    dW[-1] = 1 / m * dZ[-1].dot(A[-2].T)  # derivative of W for the last layer
    dB[-1] = 1 / m * np.sum(dZ[-1])  # derivative of B for the last layer
    # print(f"Max Bias Difference: {np.max(np.abs(dB[-1]))}")

    # other layers
    for i in range(len(Weights) - 2, -1, -1):  # from the second last layer to the first, looping backwards
        dZ[i] = Weights[i + 1].T.dot(dZ[i + 1]) * derive_ReLU(Z[i])  # derivative of Z for the current layer

        # Define whether to use the previous layer's calculated weights or the input layer's weights
        tmp = None
        if i == 0:  # if the layer previous to the current layer is the input layer
            tmp = X  # input layer
        else:
            tmp = A[i - 1]  # previous layer

        dW[i] = 1 / m * dZ[i].dot(tmp.T)  # derivative of W for the current layer
        dB[i] = 1 / m * np.sum(dZ[i])  # derivative of B for the current layer

    return dW, dB


def update_params(dW, dB, Weights, Biases):
    global Learning_Current
    for i in range(len(Weights)):
        # print(f"\n\nRunning for layer: {i}\nWeights {i} shape: {Weights[i].shape}\nBiases {i} shape: {Biases[i].shape}"
        #       f"\n dW {i} shape: {dW[i].shape}\n dB {i}: {dB[i]}")
        Weights[i] -= Learning_Current * dW[i]
        Biases[i] -= Learning_Current * dB[i]

    Learning_Current *= LEARNING_DECAY
    return Weights, Biases


def get_predictions(A):
    return np.argmax(A, 0)


def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, layers=None, Weights=None, Biases=None):
    if iterations < 1:
        return

    # Train the network using gradient descent
    if layers is None:
        layers = [784, 6, 6, 6, 10]

    if Weights is None or Biases is None:
        Weights, Biases = init_params(layers)

    for i in range(iterations):
        A, Z = forward_prop(X, Weights, Biases)
        dW, dB = back_prop(A, Z, Weights, X, Y)
        Weights, Biases = update_params(dW, dB, Weights, Biases)

        if i % math.ceil(iterations / 25) == 0:  # print every 1/100th of the iterations
            predictions = get_predictions(A[-1])
            print(f"\nIteration: {i}\nAccuracy: {get_accuracy(predictions, Y) * 100}%")

    print(f"Final Accuracy: {get_accuracy(get_predictions(A[-1]), Y) * 100}%")
    return Weights, Biases


# HANDLING LOADING/SAVING WEIGHTS AND BIASES
def load_network():
    print("Loading network...")
    return np.load(FILE_PATH, allow_pickle=True)


def save_network(data):
    print("Saving network...")
    np.save(FILE_PATH, data)


def train():
    Weights, Biases = [None] * 2

    if LOAD_FILE:
        data = load_network()
        Weights, Biases = data[0], data[1]

    Weights, Biases = gradient_descent(TRAINING_DATA, TRAINING_LABELS, TRAINING_ITERATIONS, layers=LAYERS,
                                       Weights=Weights, Biases=Biases)

    if SAVE_FILE_AFTER_TRAINING:
        data = [Weights, Biases]
        save_network(data)

    print("Finished training")
    return Weights, Biases


def process_input(input, Weights=None, Biases=None):
    if Weights is None or Biases is None:
        Weights, Biases = load_network()

    predictions, _ = forward_prop(input, Weights, Biases)
    # _ is the Z values, we don't need them - we just want the predictions

    return predictions  # the predictions are for each layer, bear in mind for outputting


def test():
    Weights, Biases = load_network()
    output = make_prediction(Weights, Biases, display=False)
    return output


def run():
    if MODE == "TRAIN":
        train()
    elif MODE == "TEST":
        test()
    else:
        print(f"Network launched in mode {MODE}, waiting.")


if __name__ == "__main__":
    run()
