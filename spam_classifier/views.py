from django.shortcuts import render
from .forms import TextForm, ImmutableTextForm
from naive_bayes import NaiveBayes
from models import SpamData
from preprocess import text_to_frequencies
import numpy as np


def index(request):
	if request.method == 'POST':
		form = TextForm(request.POST)
		if form.is_valid():
			db_data = SpamData.objects.all()
			training_data = np.empty([len(db_data), 58])
			for i in range(0, len(db_data)):
				training_data[i,:] = db_data[i].get_data()

			input_vector = text_to_frequencies(form.cleaned_data['text'])
			choice = form.cleaned_data['choice']
			bayes = NaiveBayes(training_data)
			classification = bayes.classify(input_vector)
			str_class = "NOT SPAM"
			if classification == 1:
				str_class = "SPAM"

			return render(request, 'spam_classifier/results.html',{
				'input': ImmutableTextForm(request.POST),
				'isspam': str_class,
				'details': input_vector
			})

	else:
		form = TextForm()

	return render(request, 'spam_classifier/index.html', {'form': form})