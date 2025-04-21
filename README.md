# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1..Import the standard libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

 


## Program:

Program to implement the simple linear regression model for predicting the marks scored.

~~~
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv(r"C:\Users\admin\Downloads\Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored-main\Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored-main\student_scores.csv",encoding='latin-1')

print(df)

df.head(0)

df.tail(0)

x = df.iloc[:,:-1].values

print(x)

y = df.iloc[:,1].values

print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print(y_pred)

print(y_test)

mae = mean_absolute_error(y_test,y_pred)

print("MAE: ",mae)

mse = mean_squared_error(y_test,y_pred)

print("MSE: ",mse)

rmse = np.sqrt(mse)

print("RMSE: ",rmse)

plt.scatter(x_train,y_train)

plt.plot(x_train,regressor.predict(x_train) , color ='blue')

plt.title("Hours vs Scores(training set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

plt.scatter(x_test,y_test)

plt.plot(x_test,regressor.predict(x_test),color = 'black')

plt.title("Hours vs Scores(testing set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()
~~~
~~~
Developed by: GURUPARAN G
RegisterNumber:212224220030  

~~~
## Output:


![Screenshot 2025-04-21 024150](https://github.com/user-attachments/assets/a388d684-9493-4646-961b-3c534beea0ca)

![Screenshot 2025-04-21 024204](https://github.com/user-attachments/assets/f36cd9bf-6a42-4eab-b056-8687777d8f38)

![Screenshot 2025-04-21 024222](https://github.com/user-attachments/assets/1a4de4fe-57de-42ff-808c-bf6b424faf24)

![Screenshot 2025-04-21 024238](https://github.com/user-attachments/assets/5674c476-5c75-494c-9587-8d2011a88ea8)

![Screenshot 2025-04-21 024320](https://github.com/user-attachments/assets/175aa03f-c229-4616-b6d9-989601fb00e3)

![Screenshot 2025-04-21 024341](https://github.com/user-attachments/assets/abccb32d-115f-4017-accd-035ea27643d9)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
