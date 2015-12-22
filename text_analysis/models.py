from django.db import models
from custom_fields import SerializedDataField


# DataModel that holds a numpy array
# with word and char frequencies
class SpamData(models.Model):
	spam_data = SerializedDataField()

	def get_data(self):
		return self.spam_data


# DataModel that holds the words to analyze
class IdentifierWord(models.Model):
	word = models.CharField(max_length=30)

	def get_word(self):
		return self.word
