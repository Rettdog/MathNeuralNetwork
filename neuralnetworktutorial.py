import numpy as np
import math

np.random.seed(0)

#Modeling a single neuron
#3 inputs

inputs = [1, 2, 3]
weights = [0.5, -1, 0.2]
bias = 2

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
# print(output)


#Modeling a single layer
#4 inputs & 4 neurons

inputs = [1, 2, 3, 4]
weights1 = [0.5, 1, 0.2, 2]
weights2 = [0.75, -1, 2, 5]
weights3 = [0.25, 0.1, 0.12, 0.2]
bias1 = 2
bias2 = 1
bias3 = 0.5

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + bias3,]
# print(output)

#Using NumPy
inputs = [1, 2, 3, 4]

weights =  [[0.5, 1, 0.2, 2],
            [0.75, -1, 2, 5],
            [0.25, 0.1, 0.12, 0.2]]

biases = [2,1,0.5]

output = np.dot(weights, inputs) + biases
# print(output)

#Using Batches & 2 layers

inputs =   [[1, 2, 3, 4],
            [0.1, -2.2, 0.1, 0.2],
            [0.4, 2.5, 3.1, -0.5]]

weights =  [[0.5, 1, 0.2, 2],
            [0.75, -1, 2, 5],
            [0.25, 0.1, 0.12, 0.2]]

biases = [2,1,0.5]

weights2 =  [[0.1, -0.14, 0.5],
            [0.5, -0.12, 0.33],
            [0.4, 2, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
# print(layer2_outputs)

#Using an object

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

X = [1, 2, 3, 2.5]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases

layer1 = Layer_Dense(4,5)
#must have number of inputs as neurons from previous layer
layer2 = Layer_Dense(5,2)

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)
# print(layer2.output)

#Recitified Linear Activation Function

inputs = [0,2,-1,3.3, -2.7, 1.1, 2.2, -100]
outputs = []

# for i in inputs:
#     if i>0:
#         output.append(i)
#     elif i<=0:
#         output.append(0)

# for i in inputs:
#     output.append(max(0,i))

# print(output)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

#Softmax Activation Function

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)
# print(exp_values)

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
# print(norm_values)

# Categorical Cross-Entropy (-log)

softmax_output = [0.7, 0.1, 0.2]
target_output = [1,0,0]

loss = -(math.log(softmax_output[0])*target_output[0]+
         math.log(softmax_output[1])*target_output[1]+
         math.log(softmax_output[2])*target_output[2])


# print(loss)

        # with NumPy

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0,1,1]

# print(softmax_outputs[[0,1,2], class_targets])

print(np.mean(-np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])))

        #for accuracy rather than loss
predictions = np.argmax(softmax_outputs,axis=1)
print(predictions)
accuracy = np.mean(predictions == class_targets)
print(f"acc: {accuracy}")
