from django.shortcuts import render
from .forms import TextForm, ImmutableTextForm
from naive_bayes import NaiveBayes
from perceptron import Perceptron
from multi_layer_perceptron import MultiLayerPerceptron
from models import SpamData
from preprocess import text_to_frequencies, normalize_data
import numpy as np
import logging

logger = logging.getLogger(__name__)


def index(request):
	if request.method == 'POST':
		form = TextForm(request.POST)

		if form.is_valid():
			db_data = SpamData.objects.all()
			training_data = np.empty([len(db_data), 58])
			for i in range(0, len(db_data)):
				training_data[i,:] = db_data[i].get_data()

			input_vector = text_to_frequencies(form.cleaned_data['text'])

			# TODO fix path
			dir = '/Users/Tim/Developer/Projects/NaiveTexter/spam_classifier/static/img'

			classification = None
			accuracy = None
			imgdir = None

			choice = form.cleaned_data['choice']

			if choice == "nb":
				logger.info("Naive Bayes")
				bayes = NaiveBayes()
				bayes.train(training_data)
				classification = bayes.classify(input_vector)
				accuracy = bayes.accuracy(training_data)

			elif choice == "per":
				logger.info("Perceptron")
				percep = Perceptron([57,1])
				accuracy_arr = percep.train(normalize_data(training_data), iterations=100, samples=2, get_accuracy_arr=True)
				classification = percep.classify(input_vector)
				imgdir = 'img/' + percep.plot(np.arange(accuracy_arr.size), accuracy_arr, dir)
				accuracy = accuracy_arr[accuracy_arr.size-1]

			elif choice == "mlper":
				logger.info("Multi-layer perceptron")
				mpercep = MultiLayerPerceptron([57, 29, 1])
				accuracy_arr = mpercep.train(normalize_data(training_data), iterations=10, samples=1, get_accuracy_arr=True)
				classification = mpercep.classify(input_vector)
				imgdir = 'img/' + mpercep.plot(np.arange(accuracy_arr.size), accuracy_arr, dir)
				accuracy = accuracy_arr[accuracy_arr.size-1]

			elif choice == "svm":
				logger.info("Support vector machine")
				pass

			else:
				logger.info("k nearest neighbor")
				pass

			accuracy = "Accuracy: " + accuracy.__str__() + "%"
			str_class = "NOT SPAM"
			if classification == 1:
				str_class = "SPAM"

			return render(request, 'spam_classifier/results.html', {
				'input': ImmutableTextForm(request.POST),
				'isspam': str_class,
				'details': accuracy,
				'img': imgdir
			})

	else:
		form = TextForm()

	return render(request, 'spam_classifier/index.html', {'form': form})