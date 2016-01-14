from abc import ABCMeta, abstractmethod
import numpy as np


class Classifier:
	"""
	This class is a base class for all machine-learning classes (to come).
	The class defines basic methods that should appy to all machine-learning technologies
	for identifying spam messages.
	"""
	__metaclass__ = ABCMeta
	training_data = None

	@abstractmethod
	def __init__(self, training_data):
		"""
		Constructor
		:param training_data: A matrix with training data
		:return: Nothing
		"""
		self.training_data = training_data

	@abstractmethod
	def classify(self, input_vector):
		"""
		This method classifies the datapoint specified in the constructor.
		The classifier uses the previously classified data also specified in the constructor.
		:param input_vector: the input vector to be classified
		:return: 1 if spam 0 if not spam
		"""
		pass

	@staticmethod
	def split_data_set(dataset, splitratio=0.67):
		"""
		Randomly divide a dataset into a test and a validation set
		:param dataset: the dataset to split
		:param splitratio: the ratio of which to split the dataset
		:return: a tuple (trainset, validationset)
		"""
		permutated_data = np.random.permutation(dataset)
		train = permutated_data[0:(splitratio*permutated_data[:,0].size), :]
		validate = permutated_data[(splitratio*permutated_data[:,0].size):, :]
		return train, validate

	@staticmethod
	def _separate_data_classes(dataset):
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
	def _normalize_data(data):
		pass