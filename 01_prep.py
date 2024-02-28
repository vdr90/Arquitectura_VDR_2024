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


# Cargar datos
tr_data = pd.read_csv("../Data/raw/train.csv")
test_data = pd.read_csv("../Data/raw/test.csv")


# In[3]:


#Concatenar
# etiquetas 1 & 0 para train y test data 
tr_data['is_train'] = 1
test_data['is_train'] = 0

# Combinación
train_data = pd.concat([tr_data, test_data], sort=False).reset_index(drop=True)

# Nuevos dataframes
salesprice = pd.DataFrame()
indicator = pd.DataFrame()

salesprice = train_data['SalePrice']
indicator = train_data['is_train']


# In[4]:


# Quito los datos que no tienen más del 80% de los datos llenos

threshold = 0.8  
train_data.dropna(thresh=len(train_data) * threshold, axis=1, inplace = True)


# In[5]:


# Columnas numéricas y categóricas
train_data_num = train_data.select_dtypes(exclude=['object'])
train_data_cat = train_data.select_dtypes(include=['object'])


# In[6]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

numerical_features = ['OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath','TotRmsAbvGrd', 'GarageCars', 'GarageArea'] 

# Asegurando que train_data_num contenga solo las características numéricas especificadas
train_data_num = train_data_num[numerical_features]

# Lidiando con los missing values
def median_imputation(data):
    for col in data.columns:
        median = data[col].median()
        data[col].fillna(median, inplace=True)
    return data

# Calculando la mediana del train_data_num
train_data_num = median_imputation(train_data_num)

# Llenado de las variables numéricas
total = train_data_num.isnull().sum().sort_values(ascending=False)
percent = (train_data_num.isnull().sum()*100/train_data_num.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,100- percent], axis=1, keys=['Total', 'Fill rate of the Features'])
#missing_data.head(20)


# In[7]:


train_data_num['latest_contruction'] = train_data_num[['YearBuilt', 'YearRemodAdd']].max(axis=1)
train_data_num.drop(['YearBuilt', 'YearRemodAdd'], axis=1, inplace=True)

tr_data['latest_contruction'] = tr_data[['YearBuilt', 'YearRemodAdd']].max(axis=1)
tr_data.drop(['YearBuilt', 'YearRemodAdd'], axis=1, inplace=True)


# In[8]:


# Lista de 5 características seleccionadas.
selected_features = ['OverallCond', 'latest_contruction', 'TotalBsmtSF', 'GrLivArea', 'GarageArea']

# DataFrame con las 5 características seleccionadas.
train_data_num = train_data_num[selected_features]


# In[9]:


# Estandarizamos los datos en ***train_data_num*** - Esto mejora la eficiencia de los modelos de ML.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Applying StandardScaler to only numerical columns
scaler = StandardScaler()
train_data_num = pd.DataFrame(scaler.fit_transform(train_data_num), columns=train_data_num.columns)
#train_data_num.head()


# In[10]:


# Initialize df_stats with specified columns which indicate the metrics for analysis
df_stats = pd.DataFrame(columns=['column', 'Distinct_value_incl_na', 'Distinct_value_without_na', 
                                 'missing_val', '%_missing_val'])

# List to hold the data for each column
stats = []

for c in train_data_cat.columns:
    column_stats = {
        'column': c,
        'Distinct_value_incl_na': len(list(train_data_cat[c].unique())),
        'Distinct_value_without_na': int(train_data_cat[c].nunique()),
        'missing_val': train_data_cat[c].isnull().sum(),
        '%_missing_val': (train_data_cat[c].isnull().sum() / len(train_data_cat)).round(3) * 100
    }
    stats.append(column_stats)

# Convert the list of dictionaries to a DataFrame
df_stats = pd.DataFrame(stats)
#df_stats.head()


# In[11]:


# Impute the features before encoding it.

def mode_imputation(train_data_cat):
 
    for col in train_data_cat.columns:
        mode = train_data_cat[col].mode().iloc[0]
        train_data_cat[col] = train_data_cat[col].fillna(mode)
    return train_data_cat

train_data_cat = mode_imputation(train_data_cat)


# In[12]:


nominal_cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','Heating','GarageType','SaleType','SaleCondition']
ordinal_cols = ['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']


# In[13]:


#Dividimos las variables categóricas en 2, ordinales y nominales
train_ordinal = train_data_cat[ordinal_cols]
train_nominal = train_data_cat[nominal_cols]


# In[14]:


# Let's label encode the ordinal data
def label_encode(train_ordinal):
    """
    Label encoding of the categorical features
    """
    '''Create a copy of train_ordinal'''
    train_ordinal_encoded = train_ordinal.copy()
    lab_enc_dict = {}
    for col in train_ordinal_encoded:
        lab_enc_dict[col] = LabelEncoder()
        train_ordinal_encoded[col] = lab_enc_dict[col].fit_transform(train_ordinal[col])
    return train_ordinal_encoded

train_ordinal_encoded = label_encode(train_ordinal)
#train_ordinal_encoded.head()


# In[15]:


# Let's do one-hot encoding for the nominal data
def onehot_encode(train_nominal):
    train_onehot_encoded = pd.get_dummies(train_nominal[train_nominal.columns[:-1]])
    return train_onehot_encoded

train_nominal_onehot_encoded = onehot_encode(train_nominal)

# If the encoded DataFrame contains True/False, convert them to 0/1
train_nominal_onehot_encoded = train_nominal_onehot_encoded.astype(int)
#train_nominal_onehot_encoded.head()


# In[16]:


total = train_data_cat.isnull().sum().sort_values(ascending=False)
percent = (train_data_cat.isnull().sum()*100/train_data_cat.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, 100-percent], axis=1, keys=['Total', 'Fill rate of the Features'])
#missing_data.head(20)


# In[17]:


# Concatenate
train_data_cat = pd.concat([train_ordinal_encoded, train_nominal_onehot_encoded], axis=1)

# Applying StandardScaler 
scaler = StandardScaler()
train_data_cat = pd.DataFrame(scaler.fit_transform(train_data_cat), columns=train_data_cat.columns)
#train_data_cat.head()


# In[18]:


# Concatenate the numerical and categorical dataframes and add the flag for train-test data and also the target variable.
train_data_v1 = pd.concat([train_data_num, train_data_cat], axis=1)
train_data_v1['SalePrice'] = salesprice
train_data_v1['indicator'] = indicator
#train_data_v1.columns


# In[19]:


# Let's eliminate the flag column after the data segregation
train_data_v2 = train_data_v1[train_data_v1['indicator'] == 1].drop(columns=['indicator'])
test_data_v2 = train_data_v1[train_data_v1['indicator'] == 0].drop(columns=['indicator'])


# In[20]:


# Applying log transformation to SalePrice field
train_data_v2['SalePrice'] = np.log(train_data_v2['SalePrice'])


# In[21]:


#train and validation data
X = train_data_v2.drop('SalePrice', axis=1)  # Features (all columns except the target variable)
y = train_data_v2['SalePrice']               # Target variable

# Splitting the dataset into training (e.g., 70%) and validation (e.g., 30%) sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# X_train, y_train are the training data
# X_test, y_test are the testing data


# In[22]:


#Exportamos los datos limpios
X.to_csv('../Data/clean/Xs.csv', index=False)
y.to_csv('../Data/clean/Ys.csv', index=False)

