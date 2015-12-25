from __future__ import division
import numpy as np
import math as math
from data.spambase_words import words, chars


class NaiveBayes:
	text = None
	words = None
	data = None

	preprocessed = False

	analysis_arr = np.empty(58)
	classification = None

	def __init__(self, text, data):
		self.text = text
		self.data = data

	# Preprocesses the text
	def preprocess(self):
		self.words = self.text.split()

		# Calculate frequency of each word
		for i in range(0, len(words)):
			self.analysis_arr[i] = 100 * (self.text.lower().count(words[i]) / len(self.words))

		# Calculate frequency of each char
		for i in range(0, len(chars)):
			self.analysis_arr[i + len(words)] = 100 * (self.text.count(chars[i]) / len(self.text))

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
		self.analysis_arr[len(chars) + len(words)] = sum(cap_count) / len(cap_count)
		# Max length of sequence of capital letters
		self.analysis_arr[len(chars) + len(words) + 1] = max(cap_count)
		# Number of capital letters
		self.analysis_arr[len(chars) + len(words) + 2] = sum(cap_count)

		self.preprocessed = True

	# Classifies the text as spam or not spam
	def classify(self):
		if not self.preprocessed:
			self.preprocess()

		if self.classification is None:
			# Separate the spam and not spam classes
			spam, notspam = self.separate_classes(self.data)

			# Get mean and var for all attributes of both classes
			stat_spam = self.attr_statistics(spam)
			stat_notspam = self.attr_statistics(notspam)

			# Make a list of the classification classes
			datasets = [stat_spam, stat_notspam]

			self.classification = self.predict(datasets, self.analysis_arr[0:self.analysis_arr.size-1])
			self.analysis_arr[self.analysis_arr.size-1] = self.classification

		return self.classification

	# Returns the classification of the text
	def get_classification(self):
		if self.classification is None:
			self.classify()
		return self.classification

	# Just a helper method to get string
	def get_classification_str(self):
		if self.get_classification() == 1:
			return "Spam"
		else:
			return "Not Spam"

	# Returns the new data point made from the text
	def get_text_data(self):
		if self.classification is None:
			self.classify()
		return self.analysis_arr

	# Returns a two lists of separated classes
	def separate_classes(self, data):
		spam_class = []
		not_spam_class = []
		for i in range(0, data[:,0].size):
			if data[i,data[0,:].size-1] == 1:
				spam_class.append(data[i,:])
			else:
				not_spam_class.append(data[i,:])
		return np.array(spam_class), np.array(not_spam_class)

	# Returns 2d array of each attribute's mean and std
	def attr_statistics(self, data):
		attr_summaries = np.empty([data[0,:].size-1, 2])
		for i in range(0, data[0,:].size-1):
			attr_summaries[i,0] = self.mean(data[:,i])
			attr_summaries[i,1] = self.stdeviation(data[:,i])
		return attr_summaries

	# Calculates to probability of belonging to a class from the attributes mean and std using gaussian
	def gaussian_prob_dens(self, datapoint, mean, stdev):
		exponent = math.exp(-(math.pow(datapoint-mean,2)/(2*math.pow(stdev,2))))
		return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

	# Computes the probabilities that an input belongs classes
	def get_class_probabilities(self, statistics, input_vec):
		probabilities = np.ones(len(statistics))
		for j in range(0, len(statistics)):
			for i in range(0, input_vec.size):
				x = input_vec[i]
				probabilities[j] *= self.gaussian_prob_dens(x, statistics[j][i,0], statistics[j][i,1])
		return probabilities

	# Predicts which class an input vector belongs to
	def predict(self, statistics, input_vec):
		probs = self.get_class_probabilities(statistics, input_vec)
		if probs[0] > probs[1]:
			return 1
		else:
			return 0

	# Returns predictions of a data set
	def get_predictions(self, statistics, data_set):
		predictions = np.empty(data_set[:,0].size)
		for i in range(0, predictions.size):
			prediction = self.predict(statistics, data_set[i,:-1])
			predictions[i] = prediction
		return predictions

	# Splits a dataset into a train and validation set
	def split_dataset(self, dataset, splitratio=0.67):
		permutated_data = np.random.permutation(dataset)
		train = permutated_data[0:splitratio*permutated_data[:,0].size, :]
		validate = permutated_data[splitratio*permutated_data[:,0].size:, :]
		return train, validate

	# Returns the accuracy of a test set
	def get_accuracy(self):
		spam, notspam = self.separate_classes(self.data)

		# Get mean and var for all attributes of both classes
		stat_spam = self.attr_statistics(spam)
		stat_notspam = self.attr_statistics(notspam)

		# Make a list of the classification classes
		datasets = [stat_spam, stat_notspam]

		predictions = self.get_predictions(datasets, self.data)
		class_correct = 0
		for i in range(0, predictions.size):
			if predictions[i] == self.data[i, self.data[0,:].size-1]:
				class_correct += 1
		return (class_correct/self.data[:,0].size) * 100

	# Returns the mean of a dataset
	def mean(self, col):
		return np.mean(col)

	# Returns the std of a dataset
	def stdeviation(self, col):
		return np.std(col)


