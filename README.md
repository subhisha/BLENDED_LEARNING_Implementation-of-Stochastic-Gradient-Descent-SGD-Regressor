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
<img width="1016" height="737" alt="Screenshot 2026-03-28 111406" src="https://github.com/user-attachments/assets/2fb71144-cd05-4ebc-b4f1-a4627b519612" />
<img width="542" height="204" alt="Screenshot 2026-03-28 111616" src="https://github.com/user-attachments/assets/a541db74-0aaf-4a66-9c3a-16a10acdfe06" />
<img width="728" height="282" alt="image" src="https://github.com/user-attachments/assets/3471df5e-aadc-43ab-b6d8-21c07bd15b9f" />
<img width="1168" height="303" alt="Screenshot 2026-03-28 112045" src="https://github.com/user-attachments/assets/cfb49af5-10f4-4c9f-9692-953c3241fa9b" />
<img width="1295" height="702" alt="Screenshot 2026-03-28 112157" src="https://github.com/user-attachments/assets/4bd8fd8c-64fb-4166-bd5a-254900b0d059" />

## Result
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.

