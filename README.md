# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Initialize model parameters (weights and bias) with small random values.
2. For each training example, predict the output and calculate the error.
3. Update weights and bias using the gradient of the loss function:
4. Repeat the process for multiple epochs until the error is minimized and the model converges.

```
## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())

data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)
X=data.drop('price',axis=1)
y=data['price']

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1,1))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)

sgd_model.fit(X_train,y_train)
y_pred=sgd_model.predict(X_test)

print('Name: SUBHISHA P ')
print('Reg. No: 212225040431')

print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"R^2: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")

print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.grid(True)
plt.show()
```

## Output:
<img width="1055" height="764" alt="Screenshot 2026-02-24 142549" src="https://github.com/user-attachments/assets/6355109b-7d38-43a2-b5a3-4e0961cf5d11" />
<img width="627" height="217" alt="image" src="https://github.com/user-attachments/assets/4253405a-88a6-4d52-aa84-f9038630488d" />
<img width="317" height="134" alt="Screenshot 2026-03-28 111703" src="https://github.com/user-attachments/assets/2fc3241b-e878-4641-8bc3-9ea4f6583dca" />
<img width="1295" height="702" alt="image" src="https://github.com/user-attachments/assets/695b8bcb-6219-46f4-8207-b44b4ad5ad9e" />

## Result
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.

