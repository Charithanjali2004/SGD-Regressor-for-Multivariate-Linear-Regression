# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary libraries

2.Load and analyse the dataset

3.Convert the dataset into pandas dataframe for easier access

4.Go with preprocessing if required

5.Assign the input features and target variable

6.Standardize the input features using StandardScaler

7.Train the model using SGDRegressor and MultiOutputRegressor

8.Now test the model with new values

9.And measure the accuracy using MSE 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Kanamarlapudi Sai Charithanjali
RegisterNumber:  212224240069
*/

    import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    from sklearn.linear_model import SGDRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    
    data=fetch_california_housing()
    print(data)
    
    df=pd.DataFrame(data.data,columns=data.feature_names)
    df['target']=data.target
    print(df.head())
    print(df.tail())
    print(df.info())
    
    x=df.drop(columns=['AveOccup','target'])
    y=df['target']
    
    print(x.shape)
    print(y.shape)
    print(x.info())
    print(y.info())
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    
    scaler_x=StandardScaler()
    x_train=scaler_x.fit_transform(x_train)
    x_test=scaler_x.transform(x_test)
    scaler_y=StandardScaler()
    
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    y_train=scaler_y.fit_transform(y_train)
    y_test=scaler_y.transform(y_test)
    
    sgd=SGDRegressor(max_iter=1000,tol=1e-3)
    multi_output_sgd=MultiOutputRegressor(sgd)
    multi_output_sgd.fit(x_train,y_train)
    y_pred=multi_output_sgd.predict(x_test)
    y_pred=scaler_y.inverse_transform(y_pred)
    y_test=scaler_y.inverse_transform(y_test)
    
    mse=mean_squared_error(y_test,y_pred)
    print("Mean Squared Error:",mse)
    
    print("\nPredictions:\n",y_pred[:5])

```

## Output:

<img width="1855" height="314" alt="Screenshot 2025-09-01 162108" src="https://github.com/user-attachments/assets/03212ef9-973f-4c1f-a337-1d6e343cdb9f" />


<img width="1868" height="845" alt="Screenshot 2025-09-01 162223" src="https://github.com/user-attachments/assets/1a64bd5c-bc16-43c4-add3-bdd37c8af072" />


<img width="1864" height="132" alt="Screenshot 2025-09-01 162310" src="https://github.com/user-attachments/assets/d5145937-5242-441c-b102-12626b76819e" />


<img width="1859" height="580" alt="Screenshot 2025-09-01 162334" src="https://github.com/user-attachments/assets/cd0d0f6f-76d5-45d6-93af-21f5372ee90e" />


<img width="1860" height="115" alt="Screenshot 2025-09-01 162347" src="https://github.com/user-attachments/assets/7350726d-9953-4ee2-bc8e-6f81a20fc7de" />


<img width="1237" height="269" alt="Screenshot 2025-09-01 162431" src="https://github.com/user-attachments/assets/1ab86ffe-9626-4669-811d-84bec1018952" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
