import numpy as np
from data.spambase_words import words, chars
from sklearn import preprocessing


def text_to_frequencies(text):
	"""
	Preprocesses the text with extraction of capital letters, word and char frequencies.
	:param text: the text to be preprocessed
	:return: an array with extracted frequencies
	"""
	analysis_arr = np.empty(57)

	# Calculate frequency of each word
	for i in range(0, len(words)):
		analysis_arr[i] = _word_freq(text, words[i])

	# Calculate frequency of each char
	for i in range(0, len(chars)):
		analysis_arr[i + len(words)] = _char_freq(text, chars[i])

	cc = _cap_count(text)

	if len(cc) != 0:
		# Average length of sequence of capital letters
		analysis_arr[len(chars) + len(words)] = sum(cc) / len(cc)
		# Max length of sequence of capital letters
		analysis_arr[len(chars) + len(words) + 1] = max(cc)
	else:
		analysis_arr[len(chars) + len(words)] = 0
		analysis_arr[len(chars) + len(words) + 1] = 0

	# Number of capital letters
	analysis_arr[len(chars) + len(words) + 2] = sum(cc)

	return analysis_arr


def split_dataset(dataset, splitratio=0.7):
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


def normalize_data(data, ignore_class_col=True):
	"""
	Normalizes the values in an numpy array
	:param data: the data to be normalized
	:param ignore_class_col: wheter or not to ignore the last column
	i.e the class
	:return: a new normalized dataset
	"""
	if ignore_class_col:
		data[:,:-1] = preprocessing.scale(data[:,:-1])
		return data
	else:
		return preprocessing.scale(data)


def _word_freq(text, word):
	"""
	Calculates the frequency of a word in a text
	:param text: the text to check
	:param word: the word to look for
	:return: float 100 * (ocurrances / words)
	"""
	return 100 * (text.lower().count(word) / len(text.split()))


def _char_freq(text, char):
	"""
	Calculates the frequency of a char in a text
	:param text: the text to check
	:param char: the char to look for
	:return: float 100 * (ocurrances / chars)
	"""
	return 100 * (text.count(char) / len(text))


def _cap_count(text):
	"""
	Computes the num of capital letters in a row in a text
	:param text: the text to analyze
	:return: a list with no of capital letters in a row ex [4, 3]
	"""
	cap_count = []
	i = 0
	while i < (len(text)):
		if text[i].isupper():
			uppers = 1
			for j, k in enumerate(text[i+1:]):
				if k.isalpha():
					if not k.isupper():
						break
					else:
						uppers += 1
			cap_count.append(uppers)
			i += i+1
		else:
			i += 1
	return cap_count