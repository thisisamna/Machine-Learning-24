
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures



MODEL_DEGREE=2


data = pd.read_csv('assignment2dataset.csv')
data.info()



X=data.drop('Performance Index', axis=1,)#Features
Y=data['Performance Index'] #Label
#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)



le=LabelEncoder()
X_train['Extracurricular Activities'] = le.fit_transform(X_train['Extracurricular Activities'])
X_train.info()



train_data = pd.concat([X_train, y_train], axis=1)


#Get the correlation between the features
corr = train_data.corr()
#Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Performance Index'])>0.25]
#Correlation plot
top_feature = top_feature.delete(-1)
X_train = X_train[top_feature]
X.info()





def PolynomialFeatures(X, degree):
    # transforms the existing features to higher degree features.
    originalFeatures =list(X.columns)

    for i in range(degree-1):
        columns =list(X.columns)
        for feature1 in list(columns):
            for feature2 in list(originalFeatures):
                newFeature = feature1 +"*"+feature2
                newFeatureReversed = feature2 + "*" + feature1
                if newFeature not in X and newFeatureReversed not in X:
                    #X.insert(X.shape[1],newFeature,0)
                    newColumn=X.loc[:,feature1]*X.loc[:,feature2]
                    X = pd.concat([X,newColumn], axis=1)
                    X.columns.values[-1]= newFeature
    return X
X_train_poly = PolynomialFeatures(X_train, degree=MODEL_DEGREE)



X_train_poly.head()


# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)
# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

print("Model Training Error:" + str(mean_squared_error(y_train, y_train_predicted)))



#Test data preprocessing
X_test['Extracurricular Activities'] = le.fit_transform(X_test['Extracurricular Activities'])
X_test = X_test[top_feature]

X_test_poly = PolynomialFeatures(X_test, degree=MODEL_DEGREE)

# predicting on test data-set
y_test_predicted = poly_model.predict(X_test_poly)

print("Model Test Error:" + str(mean_squared_error(y_test, y_test_predicted)))






