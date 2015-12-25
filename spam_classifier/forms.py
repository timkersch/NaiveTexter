from django import forms


class TextForm(forms.Form):
	text = forms.CharField(label="", help_text="", widget=forms.Textarea(attrs={'placeholder':'Text to process...', 'cols':10}))


class ImmutableTextForm(forms.Form):
	text = forms.CharField(label="Processed text", help_text="", widget=forms.Textarea(attrs={'cols':10, 'disabled':'true'}))