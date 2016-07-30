from abc import ABCMeta, abstractmethod
import numpy as np


class Classifier:
	"""
	This class is a base class for all machine-learning classes (to come).
	The class defines basic methods that should appy to all machine-learning technologies
	for identifying spam messages.
	"""
	__metaclass__ = ABCMeta

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