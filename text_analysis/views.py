from django.shortcuts import render
from .forms import TextForm, ImmutableTextForm, ImmutableTextForm2


def index(request):
	if request.method == 'POST':
		form = TextForm(request.POST)
		if form.is_valid():
			return render(request, 'text_analysis/results.html',{
				'input': ImmutableTextForm(request.POST),
				'isspam': 'SPAM',
				'details': ImmutableTextForm2(request.POST)})

	else:
		form = TextForm()

	return render(request, 'text_analysis/index.html', {'form': form})