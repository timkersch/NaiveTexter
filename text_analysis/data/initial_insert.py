import numpy as np
from text_analysis.models import SpamData, IdentifierWord
from spambase_words import words


# Initially Insert spam data to model
def insert_spambase(filename='spambase.data'):
	spambase_data = np.loadtxt(filename, delimiter=',')
	for i in range(0, spambase_data[:, 0].size):
		SpamData(spam_data=spambase_data[i, :]).save()


# Initially insert words to model
def insert_words(list_of_words=words):
	for w in list_of_words:
		IdentifierWord(word=w).save()
