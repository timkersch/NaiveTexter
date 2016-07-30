import logging

import numpy as np
from django.shortcuts import render

from ml.decision_tree import DecisionTree
from ml.multi_layer_perceptron import MultiLayerPerceptron
from ml.naive_bayes import NaiveBayes
from models import SpamData
from spam_classifier.ml.perceptron import Perceptron
from spam_classifier.ml.preprocess import text_to_frequencies, normalize_data, split_dataset
from .forms import TextForm, ImmutableTextForm

logging.basicConfig(level=logging.INFO)
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
				accuracy = str(bayes.accuracy(training_data))

			elif choice == "per" or choice == "mlper":
				if choice == "per":
					logger.info("Perceptron")
					network = Perceptron([57,1], eta=0.1)
				else:
					logger.info("Multi-layer Perceptron")
					network = MultiLayerPerceptron([57, 17, 1], eta=0.1)

				training_data, valid_data = split_dataset(training_data)
				accuracy_train, accuracy_valid = network.train(normalize_data(training_data), normalize_data(valid_data), iterations=1000000, samples=1000)
				classification = network.classify(normalize_data(input_vector, False))
				imgdir = 'img/' + network.plot(dir, np.arange(accuracy_train.size), accuracy_train, accuracy_valid)
				accuracy = "Train: " + str(accuracy_train[accuracy_train.size-1]) + "% - Valid: " + str(accuracy_valid[accuracy_valid.size-1]) + "%"

			elif choice == "dt":
				dt = DecisionTree()
				dt.train(training_data)
				dt.print_tree()
				accuracy = ""
				classification = 1

			elif choice == "svm":
				logger.info("Support vector machine")
				pass

			else:
				logger.info("k nearest neighbor")
				pass

			accuracy = "Accuracy: " + accuracy
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