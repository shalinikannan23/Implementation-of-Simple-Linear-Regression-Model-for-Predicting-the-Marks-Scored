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
RegisterNumber:  212222240095
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print('df.head')
df.head()
print("df.tail")
df.tail()
X=df.iloc[:,:-1].values
print("Array of X")
Y=df.iloc[:,1].values
print("Array of Y")
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print("Values of Y prediction")
print("Values of Y test")
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Training Set Graph")
plt.show()
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Test Set Graph")
plt.show()
print("Values of MSE, MAE and RMSE")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mse)
rmse=np.sqrt(mse)
print("RMSE = ",mse)
*/
```

## Output:
![image](https://github.com/shalinikannan23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118656529/190d09f7-b81e-4036-9764-1a4f7478b3fc)
![image](https://github.com/shalinikannan23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118656529/0c5abdc7-38e2-4038-b239-2cc8ce5e3b7e)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
