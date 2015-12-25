from __future__ import division
import numpy as np
import math as math
from data.spambase_words import words, chars


class NaiveBayes:
	text = None
	data = None

	__preprocessed = False
	__classification = None

	__analysis_arr = np.empty(58)

	def __init__(self, text, data):
		"""
		:param text: the text to be classified
		:param data: previously classified data
		:return: nothing
		"""
		self.text = text
		self.data = data

	def get_classification_str(self):
		"""
		A helper-method that returns a string. "Spam" or "Not Spam" based on classified data
		:return: a string "Spam" or "Not Spam"
		"""
		if self._classify() == 1:
			return "Spam"
		else:
			return "Not Spam"

	def get_text_datapoint(self):
		"""
		Returns a new data point extracted from the text
		:return: a 58 element numpy array extracted from the text
		"""
		if self.__classification is None:
			self._classify()
		return self.__analysis_arr

	def get_accuracy(self):
		"""
		Test the algorithm by comparing how many of the training-set the algorithm classifies correctly
		:return: The percentage of correct classifications of the training-set
		"""
		spam, notspam = self._separate_classes(self.data)

		# Get mean and var for all attributes of both classes
		stat_spam = self._attributes_statistics(spam)
		stat_notspam = self._attributes_statistics(notspam)

		# Make a list of the classification classes
		datasets = [stat_spam, stat_notspam]

		predictions = self._get_predictions(datasets, self.data)
		class_correct = 0
		for i in range(0, predictions.size):
			if predictions[i] == self.data[i, self.data[0,:].size-1]:
				class_correct += 1
		return (class_correct/self.data[:,0].size) * 100

	def _preprocess(self):
		"""
		This method is used to preprocess the text specified in the constructor.
		The method calulates word, char and capital letter frequencies and puts them in an array
		:return: nothing. Just sets instance variables
		"""
		extracted_words = self.text.split()

		# Calculate frequency of each word
		for i in range(0, len(words)):
			self.__analysis_arr[i] = 100 * (self.text.lower().count(words[i]) / len(extracted_words))

		# Calculate frequency of each char
		for i in range(0, len(chars)):
			self.__analysis_arr[i + len(words)] = 100 * (self.text.count(chars[i]) / len(self.text))

		# Counts sequences of capital letters
		cap_count = []
		i = 0
		while i < (len(self.text)):
			if self.text[i].isupper():
				uppers = 1
				for j, k in enumerate(self.text[i+1:]):
					if k.isalpha():
						if not k.isupper():
							break
						else:
							uppers += 1
				cap_count.append(uppers)
				i += j+1
			else:
				i += 1

		# Average length of sequence of capital letters
		self.__analysis_arr[len(chars) + len(words)] = sum(cap_count) / len(cap_count)
		# Max length of sequence of capital letters
		self.__analysis_arr[len(chars) + len(words) + 1] = max(cap_count)
		# Number of capital letters
		self.__analysis_arr[len(chars) + len(words) + 2] = sum(cap_count)

		self.__preprocessed = True

	def _classify(self):
		"""
		This method classifies the datapoint specified in the constructor using a naive bayes classifier.
		The classifier uses the previously classified data also specified in the constructor.
		:return: 1 if spam 0 if not spam
		"""
		if not self.__preprocessed:
			self._preprocess()

		if self.__classification is None:
			# Separate the spam and not spam classes
			spam, notspam = self._separate_classes(self.data)

			# Get mean and var for all attributes of both classes
			stat_spam = self._attributes_statistics(spam)
			stat_notspam = self._attributes_statistics(notspam)

			# Make a list of the classification classes
			datasets = [stat_spam, stat_notspam]

			self.__classification = self._predict(datasets, self.__analysis_arr[0:self.__analysis_arr.size - 1])
			self.__analysis_arr[self.__analysis_arr.size - 1] = self.__classification

		return self.__classification

	def _separate_classes(self, data):
		"""
		Separates data classes
		:param data: the data to separate
		:return: a tuple of numpy arrays with the two classes, 0 and 1
		"""
		spam_class = []
		not_spam_class = []
		for i in range(0, data[:,0].size):
			if data[i,data[0,:].size-1] == 1:
				spam_class.append(data[i,:])
			else:
				not_spam_class.append(data[i,:])
		return np.array(spam_class), np.array(not_spam_class)

	def _attributes_statistics(self, data):
		"""
		Computes a matrix with the mean and standard deviation of each attribute
		:param data: the data to extract mean and standard deviation from
		:return: a matrix with the mean and standard deviation of each attribute - [mean, std]
		"""
		attr_summaries = np.empty([data[0,:].size-1, 2])
		for i in range(0, data[0,:].size-1):
			attr_summaries[i,0] = np.mean(data[:,i])
			attr_summaries[i,1] = np.std(data[:,i])
		return attr_summaries

	def _get_class_probabilities(self, mean_var_matrix, input_vec):
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
				probabilities[j] *= self.__gaussian_prob_dens(x, mean_var_matrix[j][i, 0], mean_var_matrix[j][i, 1])
		return probabilities

	def _predict(self, mean_var_matrix, input_vec):
		"""
		Predict which class an input vector belongs to
		:param mean_var_matrix: a matrix with the mean and standard deviation of all attributes
		:param input_vec: the input vector
		:return: 0 if classified as not spam and 1 if classified as spam
		"""
		probs = self._get_class_probabilities(mean_var_matrix, input_vec)
		if probs[0] > probs[1]:
			return 1
		else:
			return 0

	def _get_predictions(self, mean_var_matrix, data_set):
		"""
		Predicts which classes a input matrix belongs to
		:param mean_var_matrix: a matrix with the mean and standard deviation of all attributes
		:param data_set: the input matrix
		:return: a vector of predictions
		"""
		predictions = np.empty(data_set[:,0].size)
		for i in range(0, predictions.size):
			prediction = self._predict(mean_var_matrix, data_set[i, :-1])
			predictions[i] = prediction
		return predictions

	def _split_dataset(self, splitratio=0.67):
		"""
		Randomly divide a dataset into a test and a validation set
		:param splitratio: the ratio of which to split the dataset
		:return: a tuple (trainset, validationset)
		"""
		permutated_data = np.random.permutation(self.data)
		train = permutated_data[0:splitratio*permutated_data[:,0].size, :]
		validate = permutated_data[splitratio*permutated_data[:,0].size:, :]
		return train, validate

	def __gaussian_prob_dens(self, datapoint, mean, stdev):
		exponent = math.exp(-(math.pow(datapoint-mean,2)/(2*math.pow(stdev,2))))
		return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent