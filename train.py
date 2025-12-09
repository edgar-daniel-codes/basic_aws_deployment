# train.py

### MCD ITAM - Otoño 2025
### Análisis de Datos Espaciales 
### Examen Final 
### Edgar Daniel Diaz Nava (174415)

# En el presente scrip realizamos la carga y el entrenamiento de un modelo de 
# Random Forest regression para la variable objetivo mpg con los datos del  
# dataset Auto MPG del repositorio UCI Machine Learning. 

### ----------------------------------------------------------------------------
### Importamos las librerias necesarias 

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

import pickle
import matplotlib.pyplot as plt
import joblib

### ----------------------------------------------------------------------------
### Parametros y variables 

RAW_DATA = "./data/raw/"
PREP_DATA = "./data/prep/"
CLEAN_DATA = "./data/clean/"

FILE_NAME = "auto_mpg.feather"
FILE_NAME_2 = "auto_mpg.csv"

MODEL_PATH = './models/random_forest_pipeline.pkl'
MODEL_COLS_PATH = './models/model_columns.pkl'

OBJECTIVE_VAR = "mpg"
CATEGORICAL_VARS = ["origin"]
NUMERICAL_VARS = ["cylinders","displacement", 
                  "horsepower","weight","acceleration","model_year"]
DROP_VARS = ["car_name"]


### ----------------------------------------------------------------------------
### Data Preparation 

# Leemos la base de Datos 
df = pd.read_feather(CLEAN_DATA + FILE_NAME)
df.to_csv(CLEAN_DATA + FILE_NAME_2, index = False)

data_encoded = pd.get_dummies(df, columns=['origin'], drop_first=True)


# Separamos en respuesta y regresores 

# Respuesta 
y = data_encoded[OBJECTIVE_VAR]

# Regresores (tiramos la respuesta y variables no informativas)
X = data_encoded.drop(columns=[OBJECTIVE_VAR], axis = 1)
X = X.drop(columns=DROP_VARS)

# Train-test data split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


### ----------------------------------------------------------------------------
### Entrenamiento del modelo 

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


### ----------------------------------------------------------------------------
### Evaluación del modelo 

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R² : {r2:.4f}")

### ----------------------------------------------------------------------------
### Guardamos el modelo 

# Guardar modelo Y las columnas (crucial para la API)
joblib.dump(model, MODEL_PATH)
joblib.dump(X.columns.tolist(), MODEL_COLS_PATH)

