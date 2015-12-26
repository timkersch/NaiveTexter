from __future__ import division
import numpy as np
import math as math


class NaiveBayes:
	input_vector = None
	training_data = None
	training_data_stat_list = None

	__classification = None

	def __init__(self, input_vector, training_data):
		"""
		Constructor, sets instance variables and computes a class divided training data statistics list
		:param input_vector: A preprocessed text in the form of a vector
		:param training_data: A matrix with training data
		:return: Nothing
		"""
		self.input_vector = input_vector
		self.training_data = training_data

		# Separate the spam and not spam classes
		spam, notspam = NaiveBayes._separate_classes(training_data)

		# Get mean and var for all attributes of both classes
		stat_spam = NaiveBayes._attributes_statistics(spam)
		stat_notspam = NaiveBayes._attributes_statistics(notspam)

		# Make a list of the classification classes
		self.training_data_stat_list = [stat_spam, stat_notspam]

	def classify(self):
		"""
		This method classifies the datapoint specified in the constructor using a naive bayes classifier.
		The classifier uses the previously classified data also specified in the constructor.
		:return: 1 if spam 0 if not spam
		"""
		if self.__classification is None:
			self.__classification = NaiveBayes._predict(self.training_data_stat_list, self.input_vector[0:self.input_vector.size - 1])
			self.input_vector[self.input_vector.size - 1] = self.__classification
		return self.__classification

	def get_classification(self):
		"""
		A helper-method that returns a string. "Spam" or "Not Spam" based on classified data
		:return: a string "Spam" or "Not Spam"
		"""
		if self.classify() == 1:
			return "Spam"
		else:
			return "Not Spam"

	def get_input_vector(self):
		"""
		Returns a new data point that is the input vector + the classification in the last column
		:return: a 58 element numpy array extracted from the text
		"""
		if self.__classification is None:
			self.classify()
		return self.input_vector

	def get_accuracy(self):
		"""
		Test the algorithm by comparing how many of the training-set datapoints the algorithm classifies correctly
		:return: The percentage of correct classifications of the training-set
		"""
		predictions = NaiveBayes._get_predictions(self.training_data_stat_list, self.training_data)
		class_correct = 0
		for i in range(0, predictions.size):
			if predictions[i] == self.training_data[i, self.training_data[0,:].size-1]:
				class_correct += 1
		return (class_correct/self.training_data[:,0].size) * 100

	@staticmethod
	def _separate_classes(dataset):
		"""
		Separates data classes
		:param dataset: the data to separate
		:return: a tuple of numpy arrays with the two classes, 0 and 1
		"""
		spam_class = []
		not_spam_class = []
		for i in range(0, dataset[:, 0].size):
			if dataset[i, dataset[0, :].size-1] == 1:
				spam_class.append(dataset[i, :])
			else:
				not_spam_class.append(dataset[i, :])
		return np.array(spam_class), np.array(not_spam_class)

	@staticmethod
	def _attributes_statistics(dataset):
		"""
		Computes a matrix with the mean and standard deviation of each attribute
		:param dataset: the data to extract mean and standard deviation from
		:return: a matrix with the mean and standard deviation of each attribute - [mean, std]
		"""
		attr_summaries = np.empty([dataset[0, :].size - 1, 2])
		for i in range(0, dataset[0, :].size-1):
			attr_summaries[i,0] = np.mean(dataset[:, i])
			attr_summaries[i,1] = np.std(dataset[:, i])
		return attr_summaries

	@staticmethod
	def _get_class_probabilities(mean_var_matrix, input_vec):
		"""
		Computes the probabilities that a input vector belongs to classes
		:param mean_var_matrix: a matric with the mean and standard deviation of all attributes
		:param input_vec: the input vector
		:return: a numpy array with with the probability that the input vector belongs to each class
		"""
		probabilities = np.ones(len(mean_var_matrix))
		for j in range(0, len(mean_var_matrix)):
			for i in range(0, input_vec.size):
				x = input_vec[i]
				probabilities[j] *= NaiveBayes.__gaussian_prob_dens(x, mean_var_matrix[j][i, 0], mean_var_matrix[j][i, 1])
		return probabilities

	@staticmethod
	def _predict(mean_var_matrix, input_vec):
		"""
		Predict which class an input vector belongs to
		:param mean_var_matrix: a matrix with the mean and standard deviation of all attributes
		:param input_vec: the input vector
		:return: 0 if classified as not spam and 1 if classified as spam
		"""
		probs = NaiveBayes._get_class_probabilities(mean_var_matrix, input_vec)
		if probs[0] > probs[1]:
			return 1
		else:
			return 0

	@staticmethod
	def _get_predictions(mean_var_matrix, data_set):
		"""
		Predicts which classes a input matrix belongs to
		:param mean_var_matrix: a matrix with the mean and standard deviation of all attributes
		:param data_set: the input matrix
		:return: a vector of predictions
		"""
		predictions = np.empty(data_set[:,0].size)
		for i in range(0, predictions.size):
			prediction = NaiveBayes._predict(mean_var_matrix, data_set[i, :-1])
			predictions[i] = prediction
		return predictions

	@staticmethod
	def _split_dataset(dataset, splitratio=0.67):
		"""
		Randomly divide a dataset into a test and a validation set
		:param splitratio: the ratio of which to split the dataset
		:return: a tuple (trainset, validationset)
		"""
		permutated_data = np.random.permutation(dataset)
		train = permutated_data[0:splitratio*permutated_data[:,0].size, :]
		validate = permutated_data[splitratio*permutated_data[:,0].size:, :]
		return train, validate

	@staticmethod
	def __gaussian_prob_dens(datapoint, mean, stdev):
		exponent = math.exp(-(math.pow(datapoint-mean,2)/(2*math.pow(stdev,2))))
		return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent