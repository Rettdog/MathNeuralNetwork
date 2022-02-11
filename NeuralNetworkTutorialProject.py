import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data, vertical_data  # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c

nnfs.init()


class Network:
    def __init__(self, num_inputs, num_layers, num_neurons, num_outputs):


        self.activation_relu = Activation_ReLU()
        self.activation_softmax = Activation_Softmax()
        self.loss_function = Loss_CategoricalCrossentropy()

        self.layers = []
        # print(type(self.layers))

        if num_layers == 1:

            self.layers.append(Layer_Dense(num_inputs, num_outputs))
        else:
            for i in range(num_layers):
                if i == 0:
                    print(f"adding: {(num_inputs, num_neurons)}")
                    self.layers.append(Layer_Dense(num_inputs, num_neurons))
                elif i == num_layers-1:
                    print(f"adding: {(num_neurons, num_outputs)}")
                    self.layers.append(Layer_Dense(num_neurons, num_outputs))
                else:
                    print(f"adding: {(num_neurons, num_neurons)}")
                    self.layers.append(Layer_Dense(num_neurons, num_neurons))

    def forward(self, inputs):

        values = inputs
        for layer in self.layers:
            values = layer.forward(values)
            values = self.activation_relu.forward(values)
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
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases
        return self.output

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)
        return self.output

class Activation_Softmax:
    def forward(self, inputs):
        # print(inputs)
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
            one_hot_targets[range(len(inputs)),targets] = 1
            error = one_hot_targets - inputs
        elif len(targets.shape) == 2:
            error = targets - inputs
        return error

def basic():

    X, y = vertical_data(samples=1000, classes=3)
    # plt.scatter(X[:,0], X[:,1])
    # plt.show()

    dense1 = Layer_Dense(2,3)
    activation1 = Activation_ReLU()

    dense2 = Layer_Dense(3,3)
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
        dense1.weights += random_scalar*np.random.randn(np.shape(dense1.weights)[0], np.shape(dense1.weights)[1])
        dense1.biases += random_scalar*np.random.randn(np.shape(dense1.biases)[0], np.shape(dense1.biases)[1])
        dense2.weights += random_scalar*np.random.randn(np.shape(dense2.weights)[0], np.shape(dense2.weights)[1])
        dense2.biases += random_scalar*np.random.randn(np.shape(dense2.biases)[0], np.shape(dense2.biases)[1])

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

    print(f"Final Accuracy: {total_correct/total_samples*100} Final Loss: {loss}")

    input("End")


def newtrain(p_samples, p_classes, p_layers, p_neurons, p_gens, p_random_scalar):

    X, y = spiral_data(samples=p_samples, classes=p_classes)

    network = Network(2, p_layers, p_neurons, p_classes)

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

        if (i/p_gens*100)%5 == 0:
            print("")
            print(f"Progress: {i/p_gens*100}%")

        for layer in network.layers:
            layer.weights += random_scalar*np.random.randn(np.shape(layer.weights)[0], np.shape(layer.weights)[1])
            layer.biases += random_scalar*np.random.randn(np.shape(layer.biases)[0], np.shape(layer.biases)[1])

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

    plt.scatter(X[:, 0], X[:,1], c=classifications,cmap="brg")
    plt.title("Predicted Classifications")
    plt.show()

    plt.scatter(X[:, 0], X[:,1], c=y,cmap="brg")
    plt.title("Actual Classifications")
    plt.show()

newtrain(100,3,4,32,100000, 0.25)
