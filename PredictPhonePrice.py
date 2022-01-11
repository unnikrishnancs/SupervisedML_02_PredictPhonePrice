from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import numpy as np
from Utility import *


#import csv file
data=pd.read_csv("phone_details_std.csv",na_values=["N/A"," "],usecols=[1,2,3,4,5,6])
print(data.head())

print("Before Pre-processing \n :",data.info())
print()


#pre-process data
data=data_preprocessing_regr(data,["Brand Name","camera (in no./mp)"],["screensize (in inches)","RAM (in GB)","battery (in mah)","Prize (in Rs.)"])
print("After Pre-processing \n :",data.info())

#define X and y
X=data.iloc[:,:-1]
print(X.head())

y=data.iloc[:,-1]
print(y[:5])

#split data into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)


#train the model
model=LinearRegression()
model.fit(X_train,y_train)

#test the model
y_pred=model.predict(X_test)

#print metrics
print("Mean Absolute Error \n: ",metrics.mean_absolute_error(y_pred,y_test))
print()
print("Root Mean Square Error \n: ",np.sqrt(metrics.mean_squared_error(y_pred,y_test)))
print()

#test on new data...Ex. Poco, Honor, 
'''
Brand Name	camera (in no./mp)	screensize (in inches)	RAM (in GB)	battery (in mah)
Realme	Triple	6.5	4	6000
'''
BrandName="Realme"
camera="Triple"
screensize="6.5"
RAM="4"
battery="6000"

predict_price(model,BrandName,camera,screensize,RAM,battery)


