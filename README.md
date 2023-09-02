# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the needed packages. 
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SHALINI.K
RegisterNumber: 212222240095
# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
*/
```

## Output:
To read csv file

![image](https://github.com/shalinikannan23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118656529/8c572323-f493-4a60-a2ab-605cad98ce09)

To Read Head and Tail Files

![image](https://github.com/shalinikannan23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118656529/ddb8c986-8a26-4cfe-b61d-3fa767da2807)

Compare Dataset

![image](https://github.com/shalinikannan23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118656529/0cd63521-445b-4816-be17-f07e07c113f6)

Predicted Value

![image](https://github.com/shalinikannan23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118656529/081453ac-bfac-4604-a9b5-447a95e5503c)

Graph For Training Set

![image](https://github.com/shalinikannan23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118656529/42b90a47-7157-4dbb-b339-469a120a1e3a)

Graph For Testing Set

![image](https://github.com/shalinikannan23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118656529/1d1f9c4f-e993-4eb0-a637-5b00b0b83b0c)

Error

![image](https://github.com/shalinikannan23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118656529/9af672d3-8322-45b6-9c36-e971d7979970)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
