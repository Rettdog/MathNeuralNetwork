import numpy as np
import nnfs
import matplotlib.pyplot as plt
# See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c
from nnfs.datasets import spiral_data, vertical_data
import flappybird
import math
# from network import Network

# nnfs.init()


class Network:
    def __init__(self, num_inputs, num_layers, num_neurons, num_outputs, randomness):

        self.activation_relu = Activation_ReLU()
        self.activation_softmax = Activation_Softmax()
        self.loss_function = Loss_CategoricalCrossentropy()

        self.layers = []
        # print(type(self.layers))

        if num_layers == 1:

            self.layers.append(Layer_Dense(
                num_inputs, num_outputs, randomness))
        else:
            for i in range(num_layers):
                if i == 0:
                    # print(f"adding: {(num_inputs, num_neurons)}")
                    self.layers.append(Layer_Dense(
                        num_inputs, num_neurons, randomness))
                elif i == num_layers-1:
                    # print(f"adding: {(num_neurons, num_outputs)}")
                    self.layers.append(Layer_Dense(
                        num_neurons, num_outputs, randomness))
                else:
                    # print(f"adding: {(num_neurons, num_neurons)}")
                    self.layers.append(Layer_Dense(
                        num_neurons, num_neurons, randomness))

    def forward(self, inputs):
        values = inputs
        for layer in self.layers:
            values = layer.forward(values)
            values = self.activation_relu.forward(values)
        # self.outputs = values
        self.outputs = self.activation_softmax.forward(values)
        return self.outputs

    def calc_loss(self, outputs, targets):
        return self.loss_function.calculate(outputs, targets)

    def calc_accuracy(self, outputs, targets):

        choosen_values = np.argmax(outputs, axis=1)

        # print("choosen")
        # print(choosen_values)

        total_correct = 0
        total_samples = len(choosen_values)
        for i in range(total_samples):
            if choosen_values[i] == targets[i]:
                total_correct += 1

        return total_correct/total_samples*100


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, randomness):
        self.weights = randomness*np.random.randn(n_inputs, n_neurons)
        # self.weights =
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases
        return self.output


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output


class Activation_Softmax:
    def forward(self, inputs):
        # print(inputs)
        # print(np.exp(inputs))
        exp_values = np.exp(inputs)-np.max(inputs, axis=1, keepdims=True)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


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


def basic():

    X, y = vertical_data(samples=1000, classes=3)
    # plt.scatter(X[:,0], X[:,1])
    # plt.show()

    dense1 = Layer_Dense(2, 3, 0.1)
    activation1 = Activation_ReLU()

    dense2 = Layer_Dense(3, 3, 0.1)
    activation2 = Activation_Softmax()

    lowest_error = 999999
    highest_accuracy = 0
    bestWeights1 = dense1.weights.copy()
    bestBiases1 = dense1.biases.copy()
    bestWeights2 = dense2.weights.copy()
    bestBiases2 = dense2.biases.copy()

    random_scalar = 0.05

    for gen in range(10000):

        # print(np.shape(dense1.weights))
        dense1.weights += random_scalar * \
            np.random.randn(np.shape(dense1.weights)[
                            0], np.shape(dense1.weights)[1])
        dense1.biases += random_scalar * \
            np.random.randn(np.shape(dense1.biases)[
                            0], np.shape(dense1.biases)[1])
        dense2.weights += random_scalar * \
            np.random.randn(np.shape(dense2.weights)[
                            0], np.shape(dense2.weights)[1])
        dense2.biases += random_scalar * \
            np.random.randn(np.shape(dense2.biases)[
                            0], np.shape(dense2.biases)[1])

        dense1.forward(X)
        activation1.forward(dense1.output)

        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        loss_function = Loss_CategoricalCrossentropy()
        loss = loss_function.calculate(activation2.output, y)

        choosen_values = np.argmax(activation2.output, axis=1)

        total_correct = 0
        total_samples = len(choosen_values)
        for i in range(total_samples):
            if choosen_values[i] == y[i]:
                total_correct += 1

        accuracy = total_correct/total_samples*100

        # loss_function = Loss_OffsetError()
        # loss = loss_function.calculate(activation2.output, y)

        # if loss < lowest_error:
        # print(f"Accuracy: {accuracy} Highest {highest_accuracy}")
        if accuracy >= highest_accuracy and loss < lowest_error:
            print(f"Loss: {loss} Lowest: {lowest_error}")
            print(f"Accuracy: {accuracy} Highest {highest_accuracy}")
            lowest_error = loss
            highest_accuracy = accuracy
            bestWeights1 = dense1.weights.copy()
            bestBiases1 = dense1.biases.copy()
            bestWeights2 = dense2.weights.copy()
            bestBiases2 = dense2.biases.copy()
        else:
            dense1.weights = bestWeights1.copy()
            dense1.biases = bestBiases1.copy()
            dense2.weights = bestWeights2.copy()
            dense2.biases = bestBiases2.copy()

    dense1.weights = bestWeights1.copy()
    dense1.biases = bestBiases1.copy()
    dense2.weights = bestWeights2.copy()
    dense2.biases = bestBiases2.copy()

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss_function = Loss_CategoricalCrossentropy()
    loss = loss_function.calculate(activation2.output, y)

    choosen_values = np.argmax(activation2.output, axis=1)

    total_correct = 0
    total_samples = len(choosen_values)
    for i in range(total_samples):
        if choosen_values[i] == y[i]:
            total_correct += 1

    print(
        f"Final Accuracy: {total_correct/total_samples*100} Final Loss: {loss}")

    input("End")


def newtrain(p_samples, p_classes, p_layers, p_neurons, p_gens, p_random_scalar):

    X, y = spiral_data(samples=p_samples, classes=p_classes)

    network = Network(2, p_layers, p_neurons, p_classes, 0.1)

    lowest_error = 999999
    highest_accuracy = 0

    bestWeights = []
    bestBiases = []

    for layer in network.layers:
        bestWeights.append(layer.weights.copy())
        bestBiases.append(layer.biases.copy())
    random_scalar = p_random_scalar
    scalar_step = random_scalar/p_gens

    for i in range(p_gens):

        if (i/p_gens*100) % 5 == 0:
            print("")
            print(f"Progress: {i/p_gens*100}%")

        for layer in network.layers:
            layer.weights += random_scalar * \
                np.random.randn(np.shape(layer.weights)[
                                0], np.shape(layer.weights)[1])
            layer.biases += random_scalar * \
                np.random.randn(np.shape(layer.biases)[
                                0], np.shape(layer.biases)[1])

        output = network.forward(X)

        loss = network.calc_loss(output, y)
        accuracy = network.calc_accuracy(output, y)

        if accuracy > highest_accuracy and loss < lowest_error:
            print(f"Loss: {loss:.4f} Lowest: {lowest_error:.4f}")
            print(f"Accuracy: {accuracy:.4f} Highest {highest_accuracy:.4f}")
            lowest_error = loss
            highest_accuracy = accuracy
            for i in range(len(bestWeights)):
                bestWeights[i] = network.layers[i].weights.copy()
                bestBiases[i] = network.layers[i].biases.copy()
        else:
            for i in range(len(bestWeights)):
                network.layers[i].weights = bestWeights[i].copy()
                network.layers[i].biases = bestBiases[i].copy()

        random_scalar -= scalar_step

    print("")
    print(f"Progress: 100%")
    print("Final Values:")
    print(f"Accuracy: {highest_accuracy:.4f}")
    print(f"Error: {lowest_error:.4f}")

    outputs = network.forward(X)

    classifications = np.argmax(outputs, axis=1)

    # print(classifications)

    plt.scatter(X[:, 0], X[:, 1], c=classifications, cmap="brg")
    plt.title("Predicted Classifications")
    plt.show()

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
    plt.title("Actual Classifications")
    plt.show()


def flappyTrain(genSize, threshold, epochs):

    random_scalar_weights = 0.01
    random_scalar_biases = 0

    # inputs: distance to next pipe, upper distance to pipe, lower distance to pipe
    networks = []

    for i in range(genSize):
        networks.append(Network(3, 5, 4, 2, 0.01))

    game = flappybird.FlappyBirdGame()

    for i in range(epochs):

        outcomes = game.botPlay(networks, True, True)

        # for network in networks:
        #     outcome = game.botPlay(network, False, False)
        #     outcomes = np.append(outcomes, [outcome])

        minScore = np.min(outcomes)
        maxScore = np.max(outcomes)
        averageScore = np.average(outcomes)

        if i % 1 == 0:
            print(
                f"Gen {i}: \nMin Score: {minScore}\nMax Score: {maxScore}\nAverage Score: {averageScore}\n")

        if i == (epochs-1):
            print(
                f"Final: {genSize} networks after {epochs} generations\nMin Score: {minScore}\nMax Score: {maxScore}\nAverage Score: {averageScore}\n")

        # print(f"all: {outcomes}")

        bestOutcomes = np.array([])
        for j in range(int(genSize*(threshold/100))):
            # outcomes[[np.argmax(outcomes)][0]] = -1
            bestOutcomes = np.append(bestOutcomes, [np.argmax(outcomes)][0])
            outcomes[[np.argmax(outcomes)][0]] = -1

        # print(f"minus best: {outcomes}")

        bestNetworks = []
        for j in range(len(networks)):
            if j in bestOutcomes:
                bestNetworks.append(networks[j])

        # print(f"best: {bestOutcomes}")

        numEach = (genSize-len(bestNetworks))//len(bestNetworks)+1
        numSum = genSize - (numEach)*len(bestNetworks)

        # print(f"Gensize: {genSize} \nbestNetworks: {len(bestNetworks)} \n{numEach}*{len(bestNetworks)}+{numSum}")

        newNetworks = []
        for j in range(numEach):
            newNetworks.extend(bestNetworks)
        newNetworks.extend(bestNetworks[:numSum])

        # print(f"New Length: {len(newNetworks)}")

        for network in newNetworks[threshold:]:
            for layer in network.layers:
                # print(layer.weights[0])
                layer.weights += random_scalar_weights * \
                    np.random.randn(np.shape(layer.weights)[
                                    0], np.shape(layer.weights)[1])
                layer.biases += random_scalar_biases * \
                    np.random.randn(np.shape(layer.biases)[
                                    0], np.shape(layer.biases)[1])
                # print(layer.weights[0])

        networks = newNetworks.copy()


# newtrain(100, 3, 4, 32, 100000, 0.25)
flappyTrain(100, 5, 10)
