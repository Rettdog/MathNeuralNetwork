import numpy as np
import nnfs
import matplotlib.pyplot as plt
import flappybird
import math
# from network import Network

# nnfs.init()


class Network:
    def __init__(self, num_inputs, num_layers, num_neurons, num_outputs, randomness):

        self.activation_relu = Activation_ReLU()
        self.activation_softmax = Activation_Softmax()
        self.activation_sigmoid = Activation_Sigmoid()
        self.loss_function = Loss_CategoricalCrossentropy()

        # self.randomness = randomness

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
        self.outputs = values
        # self.outputs = self.activation_softmax.forward(values)
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

    def copy(self):
        # print(f"Old Id: {id(self)}")
        output = Network(1, len(self.layers), 1, 1, 0)
        output.layers = []
        for i in range(len(self.layers)):
            output.layers.append(self.layers[i].copy())
        # print(f"New Id: {output}")
        return output


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, randomness):
        self.weights = 2*randomness * \
            np.random.random((n_inputs, n_neurons))-randomness
        # self.weights =
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases
        return self.output

    def copy(self):
        output = Layer_Dense(1, 1, 0)
        output.weights = self.weights
        output.biases = self.biases
        return output


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output


class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
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


def copyNetworkArray(networks):
    output = []
    for i in range(len(networks)):
        output.append(networks[i].copy())

    return output


def flappyTrain(genSize, threshold, epochs, shouldDraw, shouldSave, file='network.npy'):

    random_scalar_weights = .001
    scalar_step_weights = random_scalar_weights/epochs
    random_scalar_biases = 0.001
    scalar_step_biases = random_scalar_biases/epochs

    # inputs: distance to next pipe, upper distance to pipe, lower distance to pipe
    networks = []

    for i in range(genSize):
        networks.append(Network(3, 4, 4, 2, .1))

    # newNetworks = networks.copy()

    game = flappybird.FlappyBirdGame()

    for i in range(epochs):

        # outcomes = game.botPlay(networks, True,  True)
        outcomes = game.botPlay(networks, shouldDraw,  shouldDraw)

        # for network in networks:
        #     outcome = game.botPlay(network, False, False)
        #     outcomes = np.append(outcomes, [outcome])

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
        # print(f"all: {outcomes}")

        bestAmount = int(genSize*(threshold/100))

        # print(f"Best Amount: {bestAmount}")

        # for j in range(bestAmount):
        #     # outcomes[[np.argmax(outcomes)][0]] = -1
        #     bestOutcomes = np.append(bestOutcomes, np.argmax(outcomes))
        #     # print(outcomes[np.argmax(outcomes)])
        #     outcomes[[np.argmax(outcomes)][0]] = -1

        # print(outcomes[])
        # print(f"minus best: {outcomes}")

        # bestIndices = []

        # print(f"outcomes: {outcomes}")

        # for j in range(bestAmount):
        #     value = np.argmax(outcomes)
        #     print(value)
        #     for k in range(genSize):
        #         print(
        #             f"Checking if outcomes[k] {outcomes[k]} == value {value}")
        #         if outcomes[k] == value:
        #             print(f"adding: {value} at index: {k}")
        #             bestIndices.append(k)
        #             outcomes[k] = -1

        # print(f"Outcomes Before: {outcomes}")

        bestOutcomes = []

        for j in range(bestAmount):
            # outcomes[[np.argmax(outcomes)][0]] = -1
            bestOutcomes.append(np.argmax(outcomes))
            # print(outcomes[np.argmax(outcomes)])
            outcomes[np.argmax(outcomes)] = -1

        # print(f"Outcomes After: {outcomes}")

        # print(f"Best Outcomes: {bestOutcomes}")
        # print(f"Best Scores: {np.array(outcomes)[bestOutcomes]}")

        # for i in np.unique(bestOutcomes):
        #     bestIndices.extend(list(np.where(bestOutcomes == bestOutcomes)))

        # print(bestIndices)

        # print(bestOutcomes)

        # put into one line vvvv

        bestNetworks = []

        for j in bestOutcomes:
            # print(f"Keeping: {j}")
            bestNetworks.append(networks[j].copy())

        # for j in range(len(networks)):
        #     if j in bestOutcomes:
        #         bestNetworks.append(networks[j])

        # print(f"best: {bestOutcomes}")

        numEach = (genSize-len(bestNetworks))//len(bestNetworks)+1
        numSum = genSize - (numEach)*len(bestNetworks)

        # print(
        #     f"Gensize: {genSize} \nbestNetworks: {len(bestNetworks)} \n{numEach}*{len(bestNetworks)}+{numSum}")

        newNetworks = []
        # print(f"Should be empty: {newNetworks}")
        newNetworks.extend(copyNetworkArray(bestNetworks))
        for j in range(numEach-1):
            newNetworks.extend(copyNetworkArray(bestNetworks))
        newNetworks.extend(copyNetworkArray(bestNetworks[:numSum]))

        # for j in range(len(newNetworks)):
        #     print(id(newNetworks[j]))

        # print(f"Changed Length: {len(newNetworks[bestAmount:])}")

        # print(newNetworks[0].layers[0].biases)

        # print(
        #     f"Test Random: {2*random_scalar_weights*np.random.random((2,3))-random_scalar_weights}")

        netsChanged = 0

        # print(
        #     f"New Before Mutation = {game.botPlay(newNetworks, False,  False)}")

        for j in range(bestAmount, len(newNetworks)):
            netsChanged += 1
            # print(f"Num networks: {len(network.layers)}")

            for k in range(len(newNetworks[j].layers)):

                # print(j, k)

                # print(id(newNetworks[j].layers[k].weights))

                # addToWeights = random_scalar_weights * \
                #     np.zeros(np.shape(layer.weights))

                # layer.weights += addToWeights

                # layer.biases += random_scalar_biases * \
                #     np.zeros(np.shape(layer.biases))

                # if(j == 7 and k == 0):
                #     print(
                #         f"Eighth Before: {newNetworks[7].layers[0].weights[0]}")

                # if(j == 9 and k == 0):
                #     print(
                #         f"Tenth Before: {newNetworks[j].layers[k].weights[0]}")

                # if(j == len(newNetworks)-1 and k == 0):
                #     print(
                #         f"Last Before: {newNetworks[j].layers[k].weights[0]}")

                # print(f"Layer Before: {newNetworks[j].layers[k].weights}")

                newNetworks[j].layers[k].weights = newNetworks[j].layers[k].weights.copy() + 2*random_scalar_weights * \
                    np.random.random(np.shape(newNetworks[j].layers[k].weights)) - \
                    random_scalar_weights
                newNetworks[j].layers[k].biases = newNetworks[j].layers[k].biases.copy() + 2*random_scalar_biases * \
                    np.random.random(np.shape(newNetworks[j].layers[k].biases)) - \
                    random_scalar_biases

                # if(j == 7 and k == 0):
                #     print(
                #         f"Eighth After: {newNetworks[7].layers[0].weights[0]}")

                # if(j == 9 and k == 0):
                #     print(
                #         f"Tenth After: {newNetworks[j].layers[k].weights[0]}")

                # if(j == len(newNetworks)-1 and k == 0):
                #     print(
                #         f"Last After: {newNetworks[j].layers[k].weights[0]}")

                # print(f"Layer After: {newNetworks[j].layers[k].weights}")
                # print(
                #     f"Test Random: {2*random_scalar_weights*np.random.random((2,3))-random_scalar_weights}")
            # if(j == 7):
            #     print(
            #         f"Eighth After After: {newNetworks[7].layers[0].weights[0]}")
            # print(
            #     f"New During Mutation (Network: {netsChanged}) = {(game.botPlay(newNetworks, False,  False))}")

        # print(
        #     f"Eighth After After After: {newNetworks[7].layers[0].weights[0]}")
        # print(layer.weights[0])

        # layer.weights = 0.01 * \
        #     np.random.random(np.shape(layer.weights))
        # layer.biases = np.zeros(np.shape(layer.biases))

        # for j in range(bestAmount, len(newNetworks)):
        #     netsChanged += 1
        #     for layer in newNetworks[j].layers:

        #         addToWeights = random_scalar_weights * \
        #             np.random.standard_normal(np.shape(layer.weights))

        #         layer.weights += addToWeights

        #         layer.biases += random_scalar_biases * \
        #             np.random.standard_normal(np.shape(layer.biases))

        # print(f"Networks mutated: {netsChanged}")

        # print(f"Old = {game.botPlay(networks, True,  True)}")
        # print(
        #     f"New After Mutation = {game.botPlay(newNetworks, False,  False)}")

        # print("Check for equality")
        # print(f"Eighth: {newNetworks[7].layers[0].weights[0]}")
        # print(f"Tenth: {newNetworks[9].layers[0].weights[0]}")
        # print(f"Last: {newNetworks[-1].layers[0].weights[0]}")

        networks = newNetworks.copy()

        # random_scalar_weights -= scalar_step_weights
        # random_scalar_biases -= scalar_step_biases

        # print(random_scalar_weights)
        print("\n")

    if shouldSave:

        output = [networks[bestOutcomes[0]]]
        # print(bestOutcomes[0])

        # print(output)

        with open(file, 'wb') as f:
            np.save(f, output, allow_pickle=True)


def runNetwork(file, maxScore, shouldDraw=True):

    game = flappybird.FlappyBirdGame()

    with open(file, 'rb') as f:
        networks = [np.load(f, allow_pickle=True)]

    # print(networks)

    outcomes = game.botPlay(networks, shouldDraw, shouldDraw, cap=maxScore)

    print(f"Final Score: {outcomes[0]}")


fileName = 'test2.npy'

# flappyTrain(50, 15, 5, False, True, file=fileName)

runNetwork('test2.npy', 1000)
