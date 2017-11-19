import random
import numpy as np
import pandas as pd

#
# Shorthand:
#   "pd_" as a variable prefix means "partial derivative"
#   "d_" as a variable prefix means "derivative"
#   "_wrt_" is shorthand for "with respect to"
#   "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively
#   "hi" and "oh" are the shorthand for hidden to input and output to hidden
#
# Comment references:
#
# [1] Wikipedia article on Backpropagation
#   http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
# [2] Step by step Backpropagation Example
#   https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
#
# [3] Fundamentals of Neural Networks - Laurene V. Fausett
#

class NeuralNetwork:
    #LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None, choice_act = None, LEARNING_RATE=None, choice_wrt_weight_update = None, momentum = 0.75):
        self.num_inputs = num_inputs
        self.choice_act = choice_act #activation function
        self.LEARNING_RATE = LEARNING_RATE
        self.choice_wrt_weight_update = choice_wrt_weight_update #weight update method
        self.momentum = momentum

        self.hidden_layer = []
        for i in range(len(num_hidden)):
            self.hidden_layer.append(NeuronLayer(num_hidden[i], hidden_layer_bias[i]))
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights[0])
        self.init_weights_between_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

        self.inspect()

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer[0].neurons)):
            for i in range(self.num_inputs):
                if hidden_layer_weights[weight_num] == 0:
                    self.hidden_layer[0].neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer[0].neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_between_hidden_layer_neurons(self, hidden_layer_weights):
        """
        Creates connections between each hidden layer
        :param hidden_layer_weights:
        :return:
        """
        weight_num = 0
        for l in range(len(hidden_layer_weights)-1):
            for h in range(len(self.hidden_layer[l+1].neurons)):
                for i in range(len(self.hidden_layer[l].neurons)):
                    if not hidden_layer_weights[l+1]:
                        self.hidden_layer[l+1].neurons[h].weights.append(random.random())
                    else:
                        self.hidden_layer[l+1].neurons[h].weights.append(hidden_layer_weights[l+1][weight_num])
                    weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer[-1].neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        for layer in self.hidden_layer:
            print('-------')
            print('Hidden Layer')
            layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs, choice_act):
        #The activation will update with each pass through the network
        activation = inputs
        for layer in self.hidden_layer:
            activation = layer.feed_forward(inputs, choice_act)
        return self.output_layer.feed_forward(activation, choice_act)

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs, choice_act):
        self.feed_forward(training_inputs, choice_act)
        if self.choice_wrt_weight_update == 1:  # delta rule
            # 1. Output neuron deltas
            pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
            for o in range(len(self.output_layer.neurons)):

                # ∂E/∂zⱼ
                pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

            # 2. Hidden neuron deltas
            pd_errors_wrt_hidden_neuron_total_net_input = [];
            for i in range(len(self.hidden_layer)):
                pd_errors_wrt_hidden_neuron_total_net_input = [[0] *
                                                               len(self.hidden_layer[
                                                                       -1 - i].neurons)] + pd_errors_wrt_hidden_neuron_total_net_input;

                for h in range(len(self.hidden_layer[-1 - i].neurons)):

                    # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
                    # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
                    d_error_wrt_hidden_neuron_output = 0

                    if i == 0:

                        for o in range(len(self.output_layer.neurons)):
                            d_error_wrt_hidden_neuron_output += \
                                pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]
                    else:
                        for o in range(len(self.hidden_layer[-i].neurons)):
                            d_error_wrt_hidden_neuron_output += \
                                pd_errors_wrt_hidden_neuron_total_net_input[1][o] * \
                                self.hidden_layer[-i].neurons[o].weights[h]

                        # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
                        pd_errors_wrt_hidden_neuron_total_net_input[0][h] = d_error_wrt_hidden_neuron_output * \
                                                                            self.hidden_layer[-i - 1].neurons[
                                                                                h].calculate_pd_total_net_input_wrt_input()

            # 3. Update output neuron weights
            for o in range(len(self.output_layer.neurons)):
                for w_ho in range(len(self.output_layer.neurons[o].weights)):

                    # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                    pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                    # Δw = α * ∂Eⱼ/∂wᵢ
                    self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

            # 4. Update hidden neuron weights
            for l in range(len(self.hidden_layer)):
                for h in range(len(self.hidden_layer[l].neurons)):
                    for w_ih in range(len(self.hidden_layer[l].neurons[h].weights)):
                        # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                        pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[l][h] * \
                                              self.hidden_layer[l].neurons[h].calculate_pd_total_net_input_wrt_weight(
                                                  w_ih)

                        # Δw = α * ∂Eⱼ/∂wᵢ
                        self.hidden_layer[l].neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight
            #Δw = α * (tarj - outj) * xi
        elif self.choice_wrt_weight_update == 2: #Adaline
            d_error_wrt_output = [0] * len(self.output_layer.neurons)
            for o in range(len(self.output_layer.neurons)):
                for w_oh in range(len(self.output_layer.neurons[o].weights)):
                    d_error_wrt_output[o] = self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_oh) * self.output_layer.neurons[o].calculate_pd_error_wrt_output(training_outputs[o])
                    # Δw = α * (tarj - outj) * xi
                    self.output_layer.neurons[o].weights[w_oh] -= self.LEARNING_RATE * d_error_wrt_output[o]
        elif self.choice_wrt_weight_update == 3: #momentum
            # 1. Output neuron deltas
            pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
            for o in range(len(self.output_layer.neurons)):
                # ∂E/∂zⱼ
                pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

            # 2. Hidden neuron deltas
            pd_errors_wrt_hidden_neuron_total_net_input = [];
            for i in range(len(self.hidden_layer)):
                pd_errors_wrt_hidden_neuron_total_net_input = [[0] *
                                                               len(self.hidden_layer[
                                                                       -1 - i].neurons)] + pd_errors_wrt_hidden_neuron_total_net_input;

                for h in range(len(self.hidden_layer[-1 - i].neurons)):

                    # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
                    # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
                    d_error_wrt_hidden_neuron_output = 0

                    if i == 0:

                        for o in range(len(self.output_layer.neurons)):
                            d_error_wrt_hidden_neuron_output += \
                                pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]
                    else:
                        for o in range(len(self.hidden_layer[-i].neurons)):
                            d_error_wrt_hidden_neuron_output += \
                                pd_errors_wrt_hidden_neuron_total_net_input[1][o] * \
                                self.hidden_layer[-i].neurons[o].weights[h]

                        # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
                        pd_errors_wrt_hidden_neuron_total_net_input[0][h] = d_error_wrt_hidden_neuron_output * \
                                                                            self.hidden_layer[-i - 1].neurons[
                                                                                h].calculate_pd_total_net_input_wrt_input()
            ho_prev_weight = 0
            ih_prev_weight = 0
            # 3. Update output neuron weights
            for o in range(len(self.output_layer.neurons)):
                for w_ho in range(len(self.output_layer.neurons[o].weights)):
                    # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                    pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[
                        o].calculate_pd_total_net_input_wrt_weight(w_ho)
                    delta = -1.0 * self.LEARNING_RATE * pd_error_wrt_weight

                    # Δw
                    self.output_layer.neurons[o].weights[w_ho] += delta + self.LEARNING_RATE * ho_prev_weight
                    ho_prev_weight = delta


            # 4. Update hidden neuron weights
            for l in range(len(self.hidden_layer)):
                for h in range(len(self.hidden_layer[l].neurons)):
                    for w_ih in range(len(self.hidden_layer[l].neurons[h].weights)):
                        # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                        pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[l][h] * \
                                              self.hidden_layer[l].neurons[h].calculate_pd_total_net_input_wrt_weight(
                                                  w_ih)
                    delta = -1.0 * self.LEARNING_RATE * pd_error_wrt_weight
                    # Δw
                    self.hidden_layer[l].neurons[h].weights[w_ih] += delta + self.momentum * ih_prev_weight
                    ih_prev_weight = delta


    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs, self.choice_act)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

    def test(self, test_inputs, test_outputs):
        self.feed_forward(test_inputs, self.choice_act)

        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(test_outputs[o])

        return pd_errors_wrt_output_neuron_total_net_input.index(min(pd_errors_wrt_output_neuron_total_net_input))



class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs, choice_act):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs, choice_act))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs, choice_act):
        self.inputs = inputs
        self.choice_act = choice_act
        self.output = self.activation(self.calculate_total_net_input(), self.choice_act)
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # Apply the sigmoid, tanh or ReLU to squash the output of the neuron
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def activation(self, total_net_input, choice):
        if choice == 1:
            return 1 / (1 + np.exp(-total_net_input))
        elif choice == 2:
            return np.tanh(total_net_input)
        elif choice == 3:
            if total_net_input > 0 :
                return total_net_input
            else:
                return 0.01

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calculate_pd_total_net_input_wrt_input(self):
        if self.choice_act == 1:
            return self.output * (1 - self.output)
        elif self.choice_act == 2: #pd_tanh
            return (1-(self.output ** 2))
        elif self.choice_act == 3: #pd_ReLU
            if self.output > 0:
                return 1
            else:
                return 0.01

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

### Trial Values ###

epoch_sayisi = 5
momentum = 0.75 #temporary
hidden_layer_inputs_len = [2]
hidden_layer_weights= []
hidden_layer_bias=[0.35]
output_layer_bias=0.6
output_layer_weights = [0.5,0.5,0.5,0.5,0.5,0.5]

################## MENU ##############################
test = 15
print("Enter 1 for Iris Dataset, 2 for Seeds Dataset: ")
add=int(input())
if add == 1:
    df = pd.read_csv('iris.csv', sep=';', header=None)
else:
    df = pd.read_csv('seeds_dataset.csv', sep=';', header=None)
training_inputs_len = [df.shape[0] - test, df.shape[1]]
training_inputs = (df.loc[0:(df.shape[0] - test), df.columns != df.shape[1] - 1]).as_matrix()
train_outputs = (df.loc[0:(df.shape[0] - test), df.columns == df.shape[1] - 1]).as_matrix()
test_inputs = (df.loc[(df.shape[0] - test):df.shape[0], df.columns != df.shape[1] - 1]).as_matrix()
test_tmp_outputs = (df.loc[(df.shape[0] - test):df.shape[0], df.columns == df.shape[1] - 1]).as_matrix()
epoch_sayisi = int(input("Enter epoch count: "))
print("Selection of Activation Function \n Enter 1 for Sigmoid, 2 for Tanh, 3 for ReLU: ")
choice_act = int(input())
print("Selection of Weight Update Function \n Enter 1 for Delta Bar, 2 for Adaptive Learning, 3 for Momentum: ")
choice_wrt_weight_update = int(input())
if choice_wrt_weight_update == 3:
    momentum = float(input("Enter momentum value: "))
hidden_layer_weights_len = [int(input("hidden layer neuron count: "))]
hidden_layer_weights = [[0]* training_inputs.shape[1]*2]
hidden_layer_bias = [0.35]
output_layer_bias = 0.6
learning_rate = float(input("Enter Learning Rate: "))


############### Train #################
nn = NeuralNetwork(training_inputs.shape[1], [2], len(np.unique(train_outputs)), hidden_layer_weights=hidden_layer_weights, hidden_layer_bias=hidden_layer_bias, output_layer_weights=output_layer_weights, output_layer_bias=output_layer_bias, choice_act=choice_act, LEARNING_RATE=learning_rate, choice_wrt_weight_update = choice_wrt_weight_update, momentum = momentum)
for j in range(epoch_sayisi):
    for i in range(training_inputs.shape[0]):
        real_outputs = [-1] * len(np.unique(train_outputs))
        real_outputs[list(train_outputs[i])[0]-1] = 1
        nn.train(list(training_inputs[i]), real_outputs, nn.choice_act)
    if(j == epoch_sayisi-1 and i == df.shape[0]-test):
        print("Error in last epoch for last value: "+ str(np.round(nn.calculate_total_error([[list(training_inputs[i]), real_outputs]]), 9)))
print("output, target")
counter = 0
for i in range(test_inputs.shape[0]):
    test_outputs = [-1] * len(np.unique(train_outputs))
    test_outputs[list(test_tmp_outputs[i])[0]-1] = 1
    a = nn.test(list(test_inputs[i]), test_outputs)
    print(a, test_outputs.index(max(test_outputs)))
    if a == test_outputs.index(max(test_outputs)):
        counter += 1
print("Test Accuracy= "+ str(counter*100 / test_inputs.shape[0]))