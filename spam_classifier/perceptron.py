from __future__ import division
import numpy as np
from classifier import Classifier


class Perceptron(Classifier):
	"""
	This class is used to classify spam messages using a multilayer perceptron
	trained with the backpropagation algorithm.
	"""

	def __init__(self, training_data, network_size):
		super(Perceptron, self).__init__(training_data)

		self.network_size = network_size

		# Initial values of beta and eta
		self.beta = 0.5
		self.eta = 0.01

		# Randomly initialize biases and weights
		self.biases = [np.random.uniform(-1,1,y) for y in network_size[1:]]
		self.weights = [np.random.uniform(-0.2,0.2,[y,x]) for x, y in zip(network_size[:-1], network_size[1:])]

	def activation_function(self, b):
		pass

	def activation_prime(self, b):
		pass

	def train(self, train_data, valid_data, iters):
		pass

	def backpropagation(self, indata, expected_out):
		pass

	def classification_error(self, indata, outdata):
		pass

	def classify(self, input_vector):
		pass
