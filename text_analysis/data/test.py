import numpy as np
def main():
	print "Use it to test data-stuff"
	arr = np.array([[1,2,1],[4,5,1],[7,8,0]])
	t, v = split_dataset(arr,1)
	a, b = separate_classes(t)
	print "Class a"
	print a
	attr = class_summary(a)
	print attr

# Splits a dataset into a train and validation set
def split_dataset(dataset, splitratio=0.67):
	permutated_data = np.random.permutation(dataset)
	train = permutated_data[0:splitratio*permutated_data[:,0].size, :]
	validate = permutated_data[splitratio*permutated_data[:,0].size:, :]
	return train, validate

# Returns a two lists of separated classes
def separate_classes(data):
	spam_class = []
	not_spam_class = []
	for i in range(0, data[:,0].size):
		if data[i,data[0,:].size-1] == 1:
			spam_class.append(data[i,:])
		else:
			not_spam_class.append(data[i,:])
	return np.array(spam_class), np.array(not_spam_class)

# Returns the mean of a dataset
def mean(col):
	return np.mean(col)

# Returns the std of a dataset
def stdeviation(col):
	return np.std(col)

# Returns 2d array of each attribute's mean and std
def attr_statistics(data):
	attr_summaries = np.empty([2,data[0][:].size-1])
	for i in range(0, attr_summaries[0,:].size):
		attr_summaries[i,0] = mean(data[:,i])
		attr_summaries[i,1] = stdeviation(data[:,i])
	return attr_summaries

if __name__ == "__main__":
	main()

