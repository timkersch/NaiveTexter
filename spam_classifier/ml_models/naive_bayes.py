from __future__ import division
import numpy as np
import math as math
from classifier import Classifier


class NaiveBayes(Classifier):
	"""
	This class is used to classify spam messages using the naive bayes classifier.
	"""
	training_data_stat_list = None

	def __init__(self, training_data_stat_list = None):
		if training_data_stat_list is not None:
			self.training_data_stat_list = training_data_stat_list

	def classify(self, input_vector):
		classification = NaiveBayes.__get_prediction(self.training_data_stat_list, input_vector[0:input_vector.size - 1])
		input_vector[input_vector.size - 1] = classification
		return classification

	def accuracy(self, validation_set):
		"""
		Computes the percent of correct classifications on a set of known values.
		:param validation_set: the set to check the algorithm accuracy with
		:return: The percent of correctly classified texts in the validation set
		"""
		predictions = NaiveBayes.__get_predictions(self.training_data_stat_list, validation_set)
		class_correct = 0
		for i in range(0, predictions.size):
			if predictions[i] == validation_set[i, validation_set[0,:].size-1]:
				class_correct += 1
		return (class_correct/validation_set[:,0].size) * 100

	def train(self, training_data):
		self.training_data_stat_list = self.__statistics_list(training_data)

	@staticmethod
	def __statistics_list(dataset):
		"""
		Computes the statistics list of a dataset
		:param dataset: the dataset to be processed
		:return: a list of two numpy arrays with their attribute statistics
		"""
		# Separate the spam and not spam classes
		spam, notspam = NaiveBayes._separate_data_classes(dataset)

		# Get mean and var for all attributes of both classes
		stat_spam = NaiveBayes.__attributes_statistics(spam)
		stat_notspam = NaiveBayes.__attributes_statistics(notspam)

		# Make a list of the classification classes
		training_data_stat_list = [stat_spam, stat_notspam]

		return training_data_stat_list

	@staticmethod
	def __attributes_statistics(dataset):
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
	def __class_probabilities(mean_var_matrix, input_vec):
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
	def __get_prediction(mean_var_matrix, input_vec):
		"""
		Predict which class an input vector belongs to
		:param mean_var_matrix: a matrix with the mean and standard deviation of all attributes
		:param input_vec: the input vector
		:return: 0 if classified as not spam and 1 if classified as spam
		"""
		probs = NaiveBayes.__class_probabilities(mean_var_matrix, input_vec)
		if probs[0] > probs[1]:
			return 1
		else:
			return 0

	@staticmethod
	def __get_predictions(mean_var_matrix, data_set):
		"""
		Predicts which classes a input matrix belongs to
		:param mean_var_matrix: a matrix with the mean and standard deviation of all attributes
		:param data_set: the input matrix
		:return: a vector of predictions
		"""
		predictions = np.empty(data_set[:,0].size)
		for i in range(0, predictions.size):
			prediction = NaiveBayes.__get_prediction(mean_var_matrix, data_set[i, :-1])
			predictions[i] = prediction
		return predictions

	@staticmethod
	def __gaussian_prob_dens(datapoint, mean, stdev):
		exponent = math.exp(-(math.pow(datapoint-mean,2)/(2*math.pow(stdev,2))))
		return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent