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

#data.to_csv("test.csv",index=False)

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

# model overfit (Nokia, 5, 5MP, 2, 3000 ; Xiaomi, 5 , 5MP, 2 , 3000)
BrandName="Xiaomi"
screensize="5"
camera="5MP"
RAM="2"
battery="3000"

predict_price(model,BrandName,camera,screensize,RAM,battery)


