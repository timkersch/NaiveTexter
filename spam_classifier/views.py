from django.shortcuts import render
from .forms import TextForm, ImmutableTextForm
from naive_bayes import NaiveBayes
from models import SpamData
import numpy as np


def index(request):
	if request.method == 'POST':
		form = TextForm(request.POST)
		if form.is_valid():
			db_data = SpamData.objects.all()
			data_arr = np.empty([len(db_data), 58])
			for i in range(0, len(db_data)):
				data_arr[i,:] = db_data[i].get_data()
			bayes = NaiveBayes(form.cleaned_data['text'], data_arr)
			classification = bayes.get_classification_str()
			data = str(bayes.get_text_datapoint())

			return render(request, 'spam_classifier/results.html',{
				'input': ImmutableTextForm(request.POST),
				'isspam': classification,
				'details': data
			})

	else:
		form = TextForm()

	return render(request, 'spam_classifier/index.html', {'form': form})