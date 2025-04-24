# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries
Use numpy, pandas, and relevant modules from sklearn.

2.Load Dataset
Fetch California housing data and convert it to a DataFrame. Add target column HousingPrice.

3.Define Features & Targets

Features X: All columns except AveOccup and HousingPrice.

Targets Y: AveOccup and HousingPrice.

4.Split Dataset
Use train_test_split() with 80% training and 20% testing.

5.Scale Data
Apply StandardScaler() to normalize both X and Y.

6.Initialize and Train Model
Use SGDRegressor inside MultiOutputRegressor and fit on training data.

7.Predict & Inverse Transform
Predict with test data and apply inverse scaling to predictions.

8.Evaluate Model
Calculate Mean Squared Error (MSE) and print first 5 predictions.

## Program / Output:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Rakesh J.S
RegisterNumber: 212222230115
*/
```
    import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    from sklearn.linear_model import SGDRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler

    dataset=fetch_california_housing()
    df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
    df['HousingPrice']=dataset.target
    print(df.head())
![image](https://github.com/user-attachments/assets/93f4ad4a-3e96-444d-9b57-e2bcb2263fd5)
```
    X=df.drop(columns=['AveOccup','HousingPrice'])
    Y=df[['AveOccup','HousingPrice']]

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    scaler_X=StandardScaler()
    scaler_Y=StandardScaler()

    X_train = scaler_X.fit_transform(X_train) 
    X_test = scaler_X.transform(X_test) 
    Y_train = scaler_Y.fit_transform(Y_train) 
    Y_test = scaler_Y.transform(Y_test)
```
```
    sgd =  SGDRegressor(max_iter=1000, tol=1e-3) 

    multi_output_sgd= MultiOutputRegressor(sgd) 
    multi_output_sgd.fit(X_train, Y_train) 

    Y_pred=multi_output_sgd.predict(X_test) 

    Y_test= scaler_Y.inverse_transform(Y_pred)  
 
    mse= mean_squared_error (Y_test, Y_pred) 
    print("Mean Squared Error:", mse) 

    print("\nPredictions: \n",Y_pred[:5])
```
![image](https://github.com/user-attachments/assets/eb23df19-5ac3-42cc-bc4b-c194d1939fa6)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
