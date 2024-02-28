#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Librerías
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.m_estimate import MEstimateEncoder
from scipy import stats
from scipy.stats import norm, probplot
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


#Cargar datos
X = pd.read_csv("../Data/clean/Xs.csv")
y = pd.read_csv("../Data/clean/Ys.csv")
train_data = pd.read_csv("../Data/raw/train.csv")


# # Modelos predictivos de ML.

# ### Dividimos los datos en set de entrenamiento y validación.

# In[3]:


# Splitting the dataset into training (e.g., 70%) and validation (e.g., 30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# X_train, y_train are the training data
# X_test, y_test are the testing data


# ### Linear Regression

# In[4]:


# Create a linear regression model
model_lr = LinearRegression()

# Fit the model on the training set
model_lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model_lr.predict(X_test)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# ### Polynomial Regression

# In[5]:


# Create a PolynomialFeatures object with the degree of the polynomial
degree = 2  # We can adjust this value
poly = PolynomialFeatures(degree=degree)

# Transform the features into polynomial features
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Create a linear regression model
model = LinearRegression()

# Fit the model on the polynomial-transformed training set
model.fit(X_train_poly, y_train)

# Make predictions on the polynomial-transformed test set
y_pred = model.predict(X_test_poly)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# ### XGBOOST model

# In[6]:


# Create an XGBoost regression model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.1, random_state=42)

# Fit the model on the training set
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# # Resultados

# In[7]:


test_data_v2 = X
test_data_v2.head()


# In[12]:


predict = xgb_model.predict(test_data_v2)
y_pred_log = np.log(predict)
y_pred = np.exp(np.exp(y_pred_log))


# In[13]:


submission_df = pd.DataFrame({
    'Id': train_data['Id'],
    'SalePrice': y_pred
})


# In[14]:


submission_df.to_csv('../Data/output/submission.csv',index=False)


# In[ ]:




