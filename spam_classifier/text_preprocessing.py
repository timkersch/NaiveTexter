import numpy as np
from data.spambase_words import words, chars


def preprocess(text):
	"""
	Preprocesses the text with extraction of capital letter, word and char frequencies.
	:param text: the text to be preprocessed
	:return: an array with extracted frequencies
	"""
	analysis_arr = np.empty(58)

	# Calculate frequency of each word
	for i in range(0, len(words)):
		analysis_arr[i] = _word_freq(text, words[i])

	# Calculate frequency of each char
	for i in range(0, len(chars)):
		analysis_arr[i + len(text)] = _char_freq(text, text[i])

	cc = _cap_count(text)

	# Average length of sequence of capital letters
	analysis_arr[len(chars) + len(words)] = sum(cc) / len(cc)
	# Max length of sequence of capital letters
	analysis_arr[len(chars) + len(words) + 1] = max(cc)
	# Number of capital letters
	analysis_arr[len(chars) + len(words) + 2] = sum(cc)
	# Dummy classification value
	analysis_arr[analysis_arr.size-1] = -1

	return analysis_arr


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
			i += j+1
		else:
			i += 1
	return cap_count