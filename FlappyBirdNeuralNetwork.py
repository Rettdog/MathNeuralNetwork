import numpy as np
import nnfs
import matplotlib.pyplot as plt
import flappybird
import math


# Network Class that contains the layers in an individual neural network

class Network:
    def __init__(self, num_inputs, num_layers, num_neurons, num_outputs, randomness):

        self.activation_relu = Activation_ReLU()
        self.activation_softmax = Activation_Softmax()
        self.activation_sigmoid = Activation_Sigmoid()
        self.loss_function = Loss_CategoricalCrossentropy()

        self.layers = []

        # Create layers
        if num_layers == 1:

            self.layers.append(Layer_Dense(
                num_inputs, num_outputs, randomness))
        else:
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(Layer_Dense(
                        num_inputs, num_neurons, randomness))
                elif i == num_layers-1:
                    self.layers.append(Layer_Dense(
                        num_neurons, num_outputs, randomness))
                else:
                    self.layers.append(Layer_Dense(
                        num_neurons, num_neurons, randomness))

    # Feed data forward through neural network
    def forward(self, inputs):
        values = inputs
        for layer in self.layers:
            values = layer.forward(values)
            values = self.activation_relu.forward(values)
        self.outputs = values
        return self.outputs

    # Call calculate on loss function (not used)
    def calc_loss(self, outputs, targets):
        return self.loss_function.calculate(outputs, targets)

    # (not used)
    def calc_accuracy(self, outputs, targets):

        choosen_values = np.argmax(outputs, axis=1)

        total_correct = 0
        total_samples = len(choosen_values)
        for i in range(total_samples):
            if choosen_values[i] == targets[i]:
                total_correct += 1

        return total_correct/total_samples*100

    # Create a copy of the network
    def copy(self):
        output = Network(1, len(self.layers), 1, 1, 0)
        output.layers = []
        for i in range(len(self.layers)):
            output.layers.append(self.layers[i].copy())
        return output

# Layer class that stores weights and biases in numpy arrays


class Layer_Dense:

    # Initialize weights and biases to random values
    def __init__(self, n_inputs, n_neurons, randomness):
        self.weights = 2*randomness * \
            np.random.random((n_inputs, n_neurons))-randomness
        self.biases = np.zeros((1, n_neurons))

    # Feed data forward through layer
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases
        return self.output

    # Create copy of this layer
    def copy(self):
        output = Layer_Dense(1, 1, 0)
        output.weights = self.weights
        output.biases = self.biases
        return output

# Rectified Linear Activation Function


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output

# Sigmoid Activation Function


class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

# Softmax Activation Function


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs)-np.max(inputs, axis=1, keepdims=True)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

# Loss superclass


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# (not used)


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelyhoods = -np.log(correct_confidences)
        return negative_log_likelyhoods

# (not used)


class Loss_OffsetError(Loss):
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = sample_losses
        return data_loss

    def forward(self, inputs, targets):
        error = 0
        if len(targets.shape) == 1:
            one_hot_targets = np.zeros(np.shape(inputs))
            one_hot_targets[range(len(inputs)), targets] = 1
            error = one_hot_targets - inputs
        elif len(targets.shape) == 2:
            error = targets - inputs
        return error

# Returns copy of a list of Networks


def copyNetworkArray(networks):
    output = []
    for i in range(len(networks)):
        output.append(networks[i].copy())

    return output

# Train networks in flappy bird


def flappyTrain(genSize, threshold, epochs, shouldDraw, shouldSave, cap=1000, file='network.npy'):

    # Random scalars
    random_scalar_weights = .001
    random_scalar_biases = 0.001

    # Initialize networks array
    networks = []
    for i in range(genSize):
        networks.append(Network(3, 4, 4, 2, .1))

    # Create game class
    game = flappybird.FlappyBirdGame()

    for i in range(epochs):

        # Run networks through game
        outcomes = game.botPlay(networks, shouldDraw,  shouldDraw, cap=cap)

        # Get and print stats for current generation
        minScore = np.min(outcomes)
        maxScore = np.max(outcomes)
        averageScore = np.average(outcomes)

        if i % 1 == 0:
            print(
                f"Gen {i}: \nMin Score: {minScore}\nMax Score: {maxScore}\nAverage Score: {averageScore}")

        if i == (epochs-1):
            print(
                f"Final: {genSize} networks after {epochs} generations\nMin Score: {minScore}\nMax Score: {maxScore}\nAverage Score: {averageScore}")
            break

        # Calculate top number of networks to keep
        bestAmount = int(genSize*(threshold/100))

        # Get indices of best networks
        bestOutcomes = []
        for j in range(bestAmount):
            bestOutcomes.append(np.argmax(outcomes))
            outcomes[np.argmax(outcomes)] = -1

        # Put the best networks in a separate array
        bestNetworks = []
        for j in bestOutcomes:
            bestNetworks.append(networks[j].copy())

        # numEach = Number of times to multiply best networks to get to genSize
        numEach = (genSize-len(bestNetworks))//len(bestNetworks)+1
        # numSum = Remainder of best networks to fill up newNetworks
        numSum = genSize - (numEach)*len(bestNetworks)

        # Copy best networks across genSize
        newNetworks = []
        newNetworks.extend(copyNetworkArray(bestNetworks))
        for j in range(numEach-1):
            newNetworks.extend(copyNetworkArray(bestNetworks))
        newNetworks.extend(copyNetworkArray(bestNetworks[:numSum]))

        # Mutate the new networks except for first copy
        for j in range(bestAmount, len(newNetworks)):
            for k in range(len(newNetworks[j].layers)):

                newNetworks[j].layers[k].weights = newNetworks[j].layers[k].weights.copy() + 2*random_scalar_weights * \
                    np.random.random(np.shape(newNetworks[j].layers[k].weights)) - \
                    random_scalar_weights
                newNetworks[j].layers[k].biases = newNetworks[j].layers[k].biases.copy() + 2*random_scalar_biases * \
                    np.random.random(np.shape(newNetworks[j].layers[k].biases)) - \
                    random_scalar_biases

        networks = newNetworks.copy()

    # Save single best network to .npy file to be run later
    if shouldSave:

        output = [networks[bestOutcomes[0]]]
        with open(file, 'wb') as f:
            np.save(f, output, allow_pickle=True)

# Run a saved network from .npy file


def runNetwork(file, maxScore, shouldDraw=True):

    # Create game class
    game = flappybird.FlappyBirdGame()

    with open(file, 'rb') as f:
        networks = [np.load(f, allow_pickle=True)]

    # Run flappy bird game with saved network
    outcomes = game.botPlay(networks, shouldDraw, shouldDraw, cap=maxScore)

    print(f"Final Score: {outcomes[0]}")

# Training Parameters:


# Number of networks in a generation
genSize = 100
# Percentage of networks in generation to keep
threshold = 15
# Number of generations to run before stopping
epochs = 10
# Should the games be shown during training?
shouldDrawTrain = True
# Should the best network be saved?
shouldSave = True
# Score cap (+1 every 5 frames)
scoreCap = 400

# Running Parameters:

# Auto cap for stopping (+1 every 5 frames)
maxScore = 1000
# Should the games be shown?
shouldDrawRun = True

# Save file
fileName = 'test2.npy'


flappyTrain(genSize, threshold, epochs, shouldDrawTrain,
            shouldSave, cap=scoreCap, file=fileName)

runNetwork(fileName, maxScore, shouldDraw=shouldDrawRun)
