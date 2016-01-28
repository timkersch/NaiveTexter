from django.db import models
try:
	import cPickle as pickle
except:
	import pickle
import base64


class SerializedDataField(models.TextField):
	# TODO remove and implement from_db_value
	__metaclass__ = models.SubfieldBase

	def to_python(self, value):
		if value is None: return
		if not isinstance(value, basestring): return value
		value = pickle.loads(base64.b64decode(value))
		return value

	def get_db_prep_save(self, value, connection, prepared=False):
		if value is None: return
		return base64.b64encode(pickle.dumps(value))

	# TODO implement
	#def from_db_value(self, value, expression, connection, context):
	#	pass