from django.shortcuts import render
from .forms import TextForm, ImmutableTextForm
from naive_bayes import NaiveBayes


def index(request):
	if request.method == 'POST':
		form = TextForm(request.POST)
		if form.is_valid():
			bayes = NaiveBayes(form.cleaned_data['text'])
			return render(request, 'text_analysis/results.html',{
				'input': ImmutableTextForm(request.POST),
				'isspam': bayes.get_classification_str(),
				'details': str(bayes.get_text_data())
			})

	else:
		form = TextForm()

	return render(request, 'text_analysis/index.html', {'form': form})