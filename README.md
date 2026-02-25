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

from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor
data = pd.read_csv("CarPrice_Assignment (1).csv")
print(data.head())
print(data.info())
data = data.drop(['CarName','car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop('price', axis=1)
y = data['price']
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1,1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Name:SUBHISHA P')
print('Reg. No:212225040143')
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"R^2: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")
print("\nModel Coeffients:")
print("Coeeficients:",sgd_model.coef_)
print('Intercept:',sgd_model.intercept_)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices using SGDregressor")
plt.plot([min(y_test),max(y_test)], [min(y_test), max(y_test)],color='red')
plt.grid(True)
plt.show()
```

## Output:
<img width="817" height="688" alt="image" src="https://github.com/user-attachments/assets/77bb6979-fcf5-4e94-a472-2fc547f1751f" />
<img width="692" height="752" alt="image" src="https://github.com/user-attachments/assets/10222047-f1e1-45f3-87b5-9e240c9a4883" />
<img width="726" height="326" alt="image" src="https://github.com/user-attachments/assets/6b98a190-1dd5-45f6-be04-e368441d1da5" />
<img width="877" height="522" alt="image" src="https://github.com/user-attachments/assets/3ae9e92b-a98f-465b-825e-0b00845b00fe" />
<img width="950" height="588" alt="image" src="https://github.com/user-attachments/assets/02875cfa-fd39-4aba-bfcc-b0db68dc94b5" />


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
