from __future__ import division
import numpy as np
from data.spambase_words import words, chars


class NaiveBayes:
	text = None
	words = None

	preprocessed = False

	analysis_arr = np.empty(58)
	classification = None

	def __init__(self, text):
		self.text = text

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

		# TODO - This is dummy for now
		self.classification = 1
		self.analysis_arr[self.analysis_arr.size-1] = self.classification

		return self.classification

	# Returns the classification of the text
	def get_classification(self):
		if self.classification is None:
			self.classify()
		return self.classification

	# Just a helper method to get string
	def get_classification_str(self):
		classification = self.get_classification()
		if classification == 1:
			return "Spam"
		else:
			return "Not Spam"

	# Returns the new data point made from the text
	def get_text_data(self):
		if self.classification is None:
			self.classify()
		return self.analysis_arr

	# Splits a dataset into a train and validation set
	def split_dataset(self, dataset, splitratio=0.67):
		permutated_data = np.random.permutation(dataset)
		train = permutated_data[0:splitratio*permutated_data[:,0].size, :]
		validate = permutated_data[splitratio*permutated_data[:,0].size:, :]
		return train, validate

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

	# Returns the mean of a dataset
	def mean(self, col):
		return np.mean(col)

	# Returns the std of a dataset
	def stdeviation(self, col):
		return np.std(col)

	# Returns 2d array of each attribute's mean and std
	def attr_statistics(self, data):
		attr_summaries = np.empty([2,data[0][:].size-1])
		for i in range(0, attr_summaries[0,:].size):
			attr_summaries[i,0] = self.mean(data[:,i])
			attr_summaries[i,1] = self.stdeviation(data[:,i])
		return attr_summaries

