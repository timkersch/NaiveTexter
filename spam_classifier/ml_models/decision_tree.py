from __future__ import division
import numpy as np
from classifier import Classifier


class DecisionTree(Classifier):

	tree = None

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
		for i in range(0, data_set[:,0].size):
			if split_function(i):
				set1.append(data_set[i,:])
			else:
				set2.append(data_set[i,:])

		return np.array(set1), np.array(set2)

	@staticmethod
	def __unique_count(data_set):
		result_count = {}
		if data_set.size != 0:
			result_list = data_set[:,-1]
			for i in range(0, result_list.size):
				result = float(result_list[i])
				if result not in result_count:
					result_count[result] = 0
				result_count[result] += 1
		return result_count

	@staticmethod
	def __gini(data_set):
		no_rows = data_set[:,0].size
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
			p = float(unique_map[key]/data_set[:,0].size)
			entropy -= p * np.log2(p)
		return entropy

	@staticmethod
	def __buildtree(data):
		if data.size == 0:
			return DecisionTree.DecisionNode()

		current_score = DecisionTree.__entropy(data_set=data)

		best_gain = 0.0
		best_criteria = None
		best_sets = None

		# For every column except result column
		for col in range(0, data[0,:].size-1):
			column_value_set = {}
			for row in range(0, data[:,0].size):
				value = data[row][col]

				# Skip if this split was already computed
				if value in column_value_set:
					continue
				else:
					set1, set2 = DecisionTree.__split_set(data, col, value)

					# Compute information gain
					p = float(set1.size)/data[:,0].size
					gain = current_score - p * DecisionTree.__entropy(set1) - (1-p) * DecisionTree.__entropy(set2)

					# Set as best if gain improved
					if (gain > best_gain and set1.size > 0 and set2.size > 0):
						best_gain = gain
						best_criteria = (col, value)
						best_sets = (set1, set2)
					column_value_set[value] = True

		if best_gain > 0:
			true_branch = DecisionTree.__buildtree(best_sets[0])
			false_branch = DecisionTree.__buildtree(best_sets[1])
			return DecisionTree.DecisionNode(column_index=best_criteria[0], matching_value=best_criteria[1],
			                                 true_node=true_branch, false_node=false_branch)
		else:
			return DecisionTree.DecisionNode(result=DecisionTree.__unique_count(data))

	@staticmethod
	def print_tree_rec(tree, indent=''):
		# Is this a leaf node?
		if tree.result is not None:
			print str(tree.result)
		else:
			# Print the criteria
			print str(tree.column_index)+':'+str(tree.matching_value)+'? '
			# Print the branches
			print indent+'T->',
			DecisionTree.print_tree_rec(tree.true_node, indent+'  ')
			print indent+'F->',
			DecisionTree.print_tree_rec(tree.false_node, indent+'  ')

	def print_tree(self):
		DecisionTree.print_tree_rec(self.tree)

	def classify(self, input_vector):
		pass

	def train(self, training_data):
		self.tree = DecisionTree.__buildtree(training_data)