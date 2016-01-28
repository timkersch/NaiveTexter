from __future__ import division
from classifier import Classifier
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Perceptron(Classifier):
	"""
	This class is used to classify spam messages using a single-layer perceptron.
	"""

	def __init__(self, network_size, weights=None, biases=None, eta=0.01, beta=0.5):
		"""
		:param network_size: The size of the network a list with no nodes in each layer [x, y, z]
		:param weights: if network has been trained before the specify weights here
		:param biases: if network has been trained before the specify biases here
		:param eta: learning rate
		:param beta: activation function parameter
		"""

		self.network_size = network_size

		# Initial values of beta and eta
		self.beta = beta
		self.eta = eta

		if weights is None and biases is None:
			# Randomly initialize biases and weights
			self.biases = [np.random.uniform(-1,1,y) for y in network_size[1:]]
			self.weights = [np.random.uniform(-0.2,0.2,[y,x]) for x, y in zip(network_size[:-1], network_size[1:])]
		else:
			self.biases = biases
			self.weights = weights

	def train(self, train_data, iterations=10000, samples=1000, get_accuracy_arr=False):
		"""
		Train the network with a set
		:param train_data: the data to train the network with
		:param iterations: number of iterations to be performed
		:param samples: number of samples of the accuracy to collect during execution
		:param get_accuracy_arr: wether or not the accuracy should be computed during execution
		:return: the accuracy of the training set after training or an array of accuracy computed during exec
		"""

		logger.info('Train with - iterations: ' + iterations.__str__() + ' - samples: ' + samples.__str__())
		if get_accuracy_arr:
			accuracy = np.empty([samples])

		for j in range(0, iterations):
			randindex = np.random.random_integers(0, train_data[:,0].size-1)
			randin = train_data[randindex, :-1]
			randout = train_data[randindex, train_data[0,:].size-1]
			self._backpropagation(randin, randout)

			if get_accuracy_arr and iterations % (iterations / samples) == 0:
				accuracy[j / (iterations/samples)] = self.accuracy(train_data)
				logger.debug('Iteration: ' + j.__str__() + ' of: ' + iterations.__str__())

		if get_accuracy_arr:
			return accuracy
		else:
			return self.accuracy(train_data)

	def train_and_validate(self, train_data, valid_data, iterations=10000, samples=1000, get_accuracy_arr=False):
		"""
		Train the network with a set and then validate it with another set
		:param train_data: the data to train the network with
		:param valid_data: the data to validate the network with
		:param iterations: number of iterations to be performed
		:param samples: number of samples of the accuracy to collect during execution
		:param get_accuracy_arr: wether or not the accuracy should be computed during execution
		:return: the accuracy of the validation set after training or an array of accuracy computed during exec
		"""

		logger.info('Train with - iterations: ' + iterations.__str__() + ' - samples: ' + samples.__str__())
		if get_accuracy_arr:
			accuracy = np.empty([samples])

		for j in range(0, iterations):
			randindex = np.random.random_integers(0, train_data[:,0].size-1)
			randin = train_data[randindex, :-1]
			randout = train_data[randindex, train_data[0,:].size-1]
			self._backpropagation(randin, randout)

			if get_accuracy_arr and iterations % (iterations / samples) == 0:
				accuracy[j / (iterations/samples)] = self.accuracy(valid_data)
				logger.debug('Iteration: ' + j.__str__() + ' of: ' + iterations.__str__())

		if get_accuracy_arr:
			return accuracy
		else:
			return self.accuracy(valid_data)

	def classify(self, invector):
		"""
		Classify the input
		:param invector: the data to be classified
		:return: 0 if not spam and 1 if spam
		"""
		return Perceptron._to_binary(self._feed_forward(invector))

	def accuracy(self, data):
		"""
		Compute the accuracy of the network on a dataset
		:param data: the data to test the network with
		:return: the percentage of correctly classified datapoints
		"""
		indata = data[:,:-1]
		outdata = data[:,data[0,:].size-1]
		recognized = 0
		for mu in range(0, outdata.size):
			if outdata[mu] == self.classify(indata[mu,:]):
				recognized += 1
		logger.info('Recognized: ' + recognized.__str__() + ' - Out of: ' + outdata.size.__str__())
		return (recognized / outdata.size) * 100

	# Feed the network with indata i.e predict the class
	def _feed_forward(self, indata):
		for bias, weight in zip(self.biases, self.weights):
			indata = self._activation_function(np.dot(weight, indata) - bias)
		return indata

	# Activation function
	def _activation_function(self, b):
		return np.tanh(self.beta * b)

	# Derivative of the activation function
	def _activation_prime(self, b):
		return self.beta * (1 - np.power(np.tanh(self.beta * b), 2))

	# The backpropagation algorithm
	def _backpropagation(self, in_vector, expected_out):
		# Propagate forward
		sum = 0
		for j in range(0, in_vector.size):
			sum += self.weights[0][0][j] * in_vector[j]
		o_vector = self._activation_function(sum - self.biases[0])
		o_prime = self._activation_prime(sum - self.biases[0])

		# Propagate backward
		delta = (expected_out - o_vector) * o_prime
		delta_bias = self.eta * delta * -1

		delta_vector = np.empty(self.weights[0].shape)
		for j in range(0, delta_vector.size):
			delta_vector[0][j] = self.eta * delta * in_vector[j]

		self.weights[0] += delta_vector
		self.biases[0] += delta_bias

	@staticmethod
	def plot(x, y, savedir):
		"""
		Generates a plot and returns the source to it
		:param x: values in x-axis
		:param y: values in y-axis
		:param savedir: the directory to save the plot
		:return: the filename
		"""
		fig = plt.figure()
		plt.plot(x, y, '-gx')
		plt.legend("Correct classifications in percent")
		src = 'plt-gen-' + datetime.datetime.now().isoformat() + '.png'
		file = savedir + '/' + src
		plt.savefig(file)
		plt.close(fig)
		logger.info('Generated plot: ' + src)
		return src

	@staticmethod
	def _to_binary(data):
		if data <= 0:
			return 0
		else:
			return 1