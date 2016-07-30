from perceptron import Perceptron
import numpy as np


class MultiLayerPerceptron(Perceptron):
	"""
	This class is used to classify spam messages using a multi-layer perceptron.
	"""

	def __init__(self, network_size, weights=None, biases=None, eta=0.01, beta=0.5):
		"""
		:param network_size: The size of the network a list with no nodes in each layer [x, y, z]
		:param weights: if network has been trained before the specify weights here
		:param biases: if network has been trained before the specify biases here
		:param eta: learning rate
		:param beta: activation function parameter
		"""
		super(MultiLayerPerceptron, self).__init__(network_size, weights, biases, eta, beta)

		self.input_size = network_size[0]
		self.hidden_size = network_size[1]
		self.out_size = network_size[2]

	# The backpropagation algorithm for one hidden layer
	def _backpropagation(self, in_vector, expected_out):
		# For indexing
		w = o = 0
		W = O = 1

		# Propagate IN -> HIDDEN
		v_vector = np.empty(self.hidden_size)
		v_prime = np.empty(self.hidden_size)
		for j in range(0, self.hidden_size):
			summ = 0
			for k in range(0, self.input_size):
				summ += self.weights[w][j][k] * in_vector[k]
			v_vector[j] = self._activation_function(summ - self.biases[o][j])
			v_prime[j] = self._activation_prime(summ - self.biases[o][j])

		# Propagate HIDDEN -> OUTPUT
		o_vector = np.empty(self.out_size)
		o_prime = np.empty(self.out_size)
		for i in range(0, self.out_size):
			summ = 0
			for j in range(0, self.hidden_size):
				summ += self.weights[W][i][j] * v_vector[j]
			o_vector[i] = self._activation_function(summ - self.biases[O][i])
			o_prime[i] = self._activation_prime(summ - self.biases[O][i])

		# Backpropagate OUTPUT -> HIDDEN
		W_delta = (expected_out - o_vector) * o_prime
		delta_output_bias = self.eta * W_delta * -1

		delta_output = np.empty(self.weights[W].shape)
		for j in range(0, self.hidden_size):
			delta_output[0][j] = self.eta * W_delta * v_vector[j]

		# Backpropagate HIDDEN -> INPUT
		w_delta = np.empty(self.hidden_size)
		for j in range(0, self.hidden_size):
			w_delta[j] = W_delta * self.weights[W][0][j] * v_prime[j]
		delta_hidden_bias = self.eta * w_delta * -1

		delta_hidden = np.empty(self.weights[w].shape)
		for j in range(0, self.hidden_size):
			for k in range(0, self.input_size):
				delta_hidden[j][k] = self.eta * w_delta[j] * in_vector[k]

		# Update weights and biases
		self.weights[w] += delta_hidden
		self.weights[W] += delta_output

		self.biases[o] += delta_hidden_bias
		self.biases[O] += delta_output_bias
