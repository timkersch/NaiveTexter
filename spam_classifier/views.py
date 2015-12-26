from django.shortcuts import render
from .forms import TextForm, ImmutableTextForm
from naive_bayes import NaiveBayes
from models import SpamData
from text_preprocessing import preprocess
import numpy as np


def index(request):
	if request.method == 'POST':
		form = TextForm(request.POST)
		if form.is_valid():
			db_data = SpamData.objects.all()
			training_data = np.empty([len(db_data), 58])
			for i in range(0, len(db_data)):
				training_data[i,:] = db_data[i].get_data()

			input_vector = preprocess(form.cleaned_data['text'])
			bayes = NaiveBayes(input_vector, training_data)
			classification = bayes.get_classification()
			data = str(bayes.get_input_vector())

			return render(request, 'spam_classifier/results.html',{
				'input': ImmutableTextForm(request.POST),
				'isspam': classification,
				'details': data
			})

	else:
		form = TextForm()

	return render(request, 'spam_classifier/index.html', {'form': form})