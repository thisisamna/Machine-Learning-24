import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
#Loading data
data = pd.read_csv('assignment1dataset.csv')
print(data.describe())
print(data.head())

X=data.drop('Performance Index', axis=1)
Y=data['Performance Index']
print(X.shape)
print(Y.shape)

#Plotting
#plt.scatter(X, Y)
#plt.xlabel('SAT', fontsize = 20)
#plt.ylabel('GPA', fontsize = 20)
#plt.show()

L = 0.0000001  # The learning Rate
epochs = 100  # The number of iterations to perform gradient descent
m=0
c=0
n = len(X) # Number of elements in X
for i in range(epochs):
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum((Y - Y_pred)* X)  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
prediction = m*X + c


plt.scatter(X, Y)
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.plot(X, prediction, color='red', linewidth = 3)
plt.show()

print('Mean Square Error', metrics.mean_squared_error(Y, prediction))

#Predict your GPA based on your SAT Score
STA_Score=int(input('Enter your SAT score: '))
y_test=m*STA_Score + c
print('Your predicted GPA is ' + str(float(y_test)))