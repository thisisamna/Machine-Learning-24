import numpy as np
import pandas as pd

data = pd.read_csv('assignment1dataset.csv')
X1=np.array(data['Hours Studied'])
X2=data['Previous Scores']
X3=data['Sleep Hours']
X4=data['Sample Question Papers Practiced']
Y=np.array(data['Performance Index'])

def calculate_error(Y,prediction):
    error = 0
    n = len(Y)
    for i in range(n):
        error += (Y[i] - prediction[i]) ** 2
    return error / n

class linear_regressor:
    def __init__(self):
        self.theta_0 = 0 #slope
        self.theta_1 = 0 #intercept

    def fit(self, X,Y, alpha=0.000002, n=300):
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
prediction = model1.predict(X1)
print("Model 1 Error:" + str(calculate_error(Y, prediction)))
#Model 2 on X2
model2 = linear_regressor()
model2.fit(X2,Y)
prediction = model2.predict(X2)
print("Model 2 Error:" + str(calculate_error(Y, prediction)))
#Model 3 on X3
model3 = linear_regressor()
model3.fit(X3,Y)
prediction = model3.predict(X3)
print("Model 3 Error:" + str(calculate_error(Y, prediction)))

#Model 4 on X4
model4 = linear_regressor()
model4.fit(X4,Y)
prediction = model4.predict(X4)
print("Model 4 Error:" + str(calculate_error(Y, prediction)))
