##############
# Question 1 #
##############

import numpy as np
import pandas as pd

retail = pd.read_csv('onlineRetail.csv', encoding = 'iso-8859-1', dtype = {'CustomerID': str})

# Do data preprocessing
# Build LDA model
# Select representative words

##############
# Question 2 #
##############

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('microchip3.txt', delimiter = ',')
x = data[:,:2]
y = data[:,2].astype(int)
color = ['red', 'blue', 'green']
plt.figure(figsize = (12, 12))
plt.xlabel('test 1 score')
plt.ylabel('test 2 score')
plt.scatter(x[y==0,0], x[y==0,1], c=color[0], marker='x', s=30)
plt.scatter(x[y==1,0], x[y==1,1], c=color[1])
plt.scatter(x[y==2,0], x[y==2,1], c=color[2], marker='^', s=30)
plt.legend(('rejected', 'accepted', 'undecided'))
plt.show()

np.random.seed(2018)
train = np.random.choice([True, False], x.shape[0], replace=True, p=[0.6, 0.4])

x_train, y_train = x[train, :], y[train]
x_test, y_test = x[~train, :], y[~train]

x2_train, y2_train = x[np.logical_and(train, y<2),:], y[np.logical_and(train, y<2)]
x2_test, y2_test = x[np.logical_and(~train, y<2),:], y[np.logical_and(~train, y<2)]

# Answer part 1 to 3 with x2_train, y2_train, x2_test, y2_test

# Answer part 4 with x_train, y_train, x_test, y_test