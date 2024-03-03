import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv('assignment1dataset.csv')
X1=data['Hours Studied']
X2=data['Previous Scores']
X3=data['Sleep Hours']
X4=data['Sample Question Papers Practiced']
#Compound features
X5= X2 + X1 + X4
Y=np.array(data['Performance Index'])

def calculate_error(Y,prediction):
    error = 0
    n = len(Y)
    for i in range(n):
        error += (Y[i] - prediction[i]) ** 2
    return error / n

class linear_regressor:
    def __init__(self):
        self.theta_0 = 1 #intercept
        self.theta_1 = 1 #slope

    def fit(self, X,Y, alpha=0.0000002, n=300):
        error_sum = 0
        error_x_sum = 0
        for i in range(n):
            for i in range(len(Y)):
                error = (Y[i] - X[i] * self.theta_1 - self.theta_0) 
                error_sum += error 
                error_x_sum += error * X[i]
            derivative_theta_0 = -2 / n * error_sum
            derivative_theta_1 = -2 / n * error_x_sum 
            temp_0 = self.theta_0 - alpha * derivative_theta_0
            temp_1 = self.theta_1 - alpha * derivative_theta_1
            self.theta_0 = temp_0
            self.theta_1 = temp_1

    def predict(self, X):
        return self.theta_1 * X + self.theta_0


#Model 1 on X1
model1 = linear_regressor()
model1.fit(X1,Y)
prediction1 = model1.predict(X1)
print("Model 1 Error:" + str(calculate_error(Y, prediction1)))
plt.scatter(X1, Y)
plt.xlabel('X', fontsize = 20)
plt.ylabel('Y', fontsize = 20)
plt.plot(X1, prediction1, color='red', linewidth = 3)
plt.show()

#Model 2 on X2
model2 = linear_regressor()
model2.fit(X2,Y)
prediction2 = model2.predict(X2)
print("Model 2 Error:" + str(calculate_error(Y, prediction2)))
plt.scatter(X2, Y)
plt.xlabel('X', fontsize = 20)
plt.ylabel('Y', fontsize = 20)
plt.plot(X2, prediction2, color='red', linewidth = 3)
plt.show()

#Model 3 on X3
model3 = linear_regressor()
model3.fit(X3,Y)
prediction3 = model3.predict(X3)
print("Model 3 Error:" + str(calculate_error(Y, prediction3)))
plt.scatter(X3, Y)
plt.xlabel('X', fontsize = 20)
plt.ylabel('Y', fontsize = 20)
plt.plot(X3, prediction3, color='red', linewidth = 3)
plt.show()
#Model 4 on X4
model4 = linear_regressor()
model4.fit(X4,Y)
prediction4 = model4.predict(X4)
print("Model 4 Error:" + str(calculate_error(Y, prediction4)))
plt.scatter(X4, Y)
plt.xlabel('X', fontsize = 20)
plt.ylabel('Y', fontsize = 20)
plt.plot(X4, prediction4, color='red', linewidth = 3)
plt.show()

#Model 5 on X5
model5 = linear_regressor()
model5.fit(X5,Y,)
prediction5 = model5.predict(X5)
print("Model 5 Error:" + str(calculate_error(Y, prediction5)))
print("Model 5 R2 Score:" + str(metrics.r2_score(Y, prediction5)))

plt.scatter(X5, Y)
plt.xlabel('X', fontsize = 20)
plt.ylabel('Y', fontsize = 20)
plt.plot(X5, prediction5, color='red', linewidth = 3)
plt.show()