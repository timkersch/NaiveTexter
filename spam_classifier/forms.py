from django import forms

CHOICES = (
    ('nb', 'Naive bayes'),
    ('per', 'Perceptron'),
    ('mlper', 'Multilayer perceptron'),
    ('svm', 'SVM'),
    ('knn', 'k-NN'),
    ('dt', 'Decision tree'),
    ('rf', 'Random Forrest')
)

class TextForm(forms.Form):
	text = forms.CharField(label="", help_text="", widget=forms.Textarea(attrs={'placeholder':'Text to process...', 'cols':10}))
	choice = forms.ChoiceField(label="Analysis method:",choices=CHOICES)

class ImmutableTextForm(forms.Form):
	text = forms.CharField(label="Processed text", help_text="", widget=forms.Textarea(attrs={'cols':10, 'disabled':'true'}))