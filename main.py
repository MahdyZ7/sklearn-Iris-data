# load the iris dataset
from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()

# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target

## suffle data
from sklearn.utils import shuffle

X, y = shuffle(X, y, random_state=41)

## split data 10 fold
X_split = np.array_split(X, 10)
y_split = np.array_split(y, 10)
##

print("Training and Testing")
print("____________________")
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
# varabils to store training and testing results
results_train = [0] * 10
results_test = [0] * 10
y_pred_test = [0] * 10
y_pred_train = [0] * 10
##two loops the outer one selects the test data and the nested loop adds the training data in one array
for i in range(10):
	X_train = []
	y_train = []
	for j in range(10):
		if (j == i):
			continue
		if (len(X_train) == 0):
			X_train = X_split[j]
			y_train = y_split[j]
		else:
			X_train = np.concatenate((X_train, X_split[j]))
			y_train = np.concatenate((y_train, y_split[j]))
    #  training is done here and testing on training data
	gnb = GaussianNB()
	gnb.fit(X_train, y_train)
	y_pred_train[i] = gnb.predict(X_train)
	results_train[i] = (metrics.accuracy_score(y_train, y_pred_train[i]) * 100)
	print("Gaussian Naive Bayes model fold : ", i)
	print("\tTraining accuracy(in %):\t\t", round(results_train[i],2))
	
	print("\tNumber of mislabeled points :\t %d/%d" %
          ((y_train != y_pred_train[i]).sum(), X_train.shape[0]))
    # predict the test data and print it
	y_pred_test[i] = gnb.predict(X_split[i])
	results_test[i] = (metrics.accuracy_score(y_split[i], y_pred_test[i]) * 100)
	print("\tTest accuracy(in %):\t\t\t",
          round(results_train[i],2))
	print("\tNumber of mislabeled points :\t %d/%d" %
          ((y_split[i] != y_pred_test[i]).sum(), X_split[i].shape[0]))
	print("*******************")
# printing mean and variance
print("The mean accuarcy for training is: ", round(np.mean(results_train),2))
print("The mean accuarcy for testing is: ", round(np.mean(results_test),2))
print()
print("The variance for training is: ", round(np.var(results_train),2))
print("The variance for testing is: ", round(np.var(results_test),2))

## The code below is the tutoral I learned from

# # splitting X and y into training and testing sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# print(len(X_train), len(X_test), len(y_train), len(y_test));

# # training the model on training set
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)

# # making predictions on the testing set
# y_pred = gnb.predict(X_test)
# # print(y_pred)

# # comparing actual response values (y_test) with predicted response values (y_pred)
# from sklearn import metrics
# print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
