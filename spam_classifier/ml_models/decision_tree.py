from __future__ import division
import numpy as np
from classifier import Classifier


class DecisionTree(Classifier):

	class DecisionNode:
		def __init__(self, column_index=-1, matching_value=None, true_node=None, false_node=None, result=None):
			self.column_index = column_index
			self.matching_value = matching_value
			self.true_node = true_node
			self.false_node = false_node
			self.result = result

	def __init__(self):
		pass

	@staticmethod
	def __split_set(data_set, column_id, split_on_value):
		split_function = None
		if isinstance(split_on_value, int) or isinstance(split_on_value, float):
			split_function = lambda row:data_set[row][column_id] >= split_on_value
		else:
			split_function = lambda row:data_set[row][column_id] == split_on_value

		set1 = []
		set2 = []
		for i in range(0, data_set[0].size):
			if split_function(i):
				set1.append(data_set[i])
			else:
				set2.append(data_set[i])

		return np.array(set1), np.array(set2)

	@staticmethod
	def __unique_count(data_set):
		result_count = {}
		for i in range(0, data_set[0].size):
			result = data_set[i][:-1]
			if result not in result_count:
				result_count[result] = 0
			result_count[result] += 1
		return result_count

	@staticmethod
	def __gini(data_set):
		no_rows = data_set[0].size
		unique_map = DecisionTree.__unique_count(data_set)
		impurity = 0
		for key in unique_map.keys():
			p1 = float(unique_map[key]/no_rows)
			for key2 in unique_map.keys():
				if key == key2:
					continue
				else:
					p2 = float(unique_map[key2]/no_rows)
					impurity += p1*p2
		return impurity

	@staticmethod
	def __entropy(data_set):
		unique_map = DecisionTree.__unique_count(data_set)
		entropy = 0.0
		for key in unique_map.keys():
			p = float(unique_map[key]/data_set[0].size)
			entropy -= p * np.log2(p)
		return entropy

	@staticmethod
	def __buildtree(data, score_function=__entropy):
		if len(data) == 0:
			return DecisionTree.DecisionNode()
		current_score = score_function(data_set=data)

		best_gain = 0.0
		best_criteria = None
		best_sets = None
		# TODO

	def classify(self, input_vector):
		pass

	def train(self, training_data):
		pass