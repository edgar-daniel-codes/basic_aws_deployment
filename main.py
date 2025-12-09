# main.py

### MCD ITAM - Otoño 2025
### Análisis de Datos Espaciales 
### Examen Final 
### Edgar Daniel Diaz Nava (174415)

# En el presente scrip configuramos la API para nuestro modelo, incluyendo tanto 
# el modulo post para generar una inferencia a partir de una petición y el modulo
# get para generar las estadisticas descriptivas mediante llamados a la API. 
# También creamos una función auxiliar para el registro de las peticiones en la base de 
# 


### ----------------------------------------------------------------------------
### Importamos las librerias necesarias 

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Union, List
import pandas as pd
import joblib
import os
import mysql.connector
from mysql.connector import Error

### ----------------------------------------------------------------------------
### Parametros y variables 

RAW_DATA = "./data/raw/"
PREP_DATA = "./data/prep/"
CLEAN_DATA = "./data/clean/"

FILE_NAME = "auto_mpg.feather"

MODEL_PATH = './models/random_forest_pipeline.pkl'
MODEL_COLS_PATH = './models/model_columns.pkl'


### ----------------------------------------------------------------------------
### Construimos clases y metodos para la API

# Iniciamos la clase de FastAPI
app = FastAPI()

# Cargargamos el modelo ya entrenado 
model = joblib.load(MODEL_PATH)
model_columns: List[str] = joblib.load(MODEL_COLS_PATH)

# Auxiliar para stats en memoria
predictions_history: List[float] = []


# Modelo Pydantic para entrada de datos 
class CarFeatures(BaseModel):
    cylinders: int
    displacement: float
    horsepower: float
    weight: float
    acceleration: float
    model_year: int
    origin: str  # "USA", "Europe", "Japan"


# Metodo POST  para generar la inferencia 
@app.post("/predict")
def predict_mpg(req: CarFeatures):
    # Preprocesamos con una funcion auxiliar definida abajo
    X = preprocess_request(req)
    pred = float(model.predict(X)[0])

    # Guardardamos en historial en memoria
    predictions_history.append(pred)

    # Guardardamos en un DB con una función auxiliar para escribir en MySQL
    insert_log_to_db(
        {
            "cylinders": int(req.cylinders) if isinstance(req.cylinders, int) else None,
            "displacement":  float(req.displacement) if isinstance(req.displacement,float) else None,
            "horsepower":  float(req.horsepower) if isinstance(req.horsepower,float) else None,
            "weight":  float(req.weight) if isinstance(req.weight,float) else None,
            "acceleration":  float(req.acceleration) if isinstance(req.acceleration,float) else None,
            "model_year":  int(req.model_year) if isinstance(req.model_year, int) else None,
            "origin":  str(req.origin) if isinstance(req.origin, str) else None,
        },
        pred
    )

    # Devuelve la prediccion
    return {"mpg_prediction": pred}

# Metodo GET para atraer los 
@app.get("/stats")
def stats():
    """
    Devuelve conteo, promedio, máximo y mínimo de predicciones hechas
    durante la vida del proceso (en memoria).
    """
    if not predictions_history:
        return {
            "count": 0,
            "mean": None,
            "max": None,
            "min": None
        }

    import numpy as np
    arr = np.array(predictions_history)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "min": float(arr.min())
    } 



### ----------------------------------------------------------------------------
### Funciones auxiliares para registro de Logs 

# Config DB usando variables de entorno (seguridad y buena practica)
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

# Funcion para insertar valores en la tabla 
def insert_log_to_db(input_data: dict, prediction: float):
    """
    Inserta un registro en la tabla logs.
    """
    if not DB_HOST:
        # Si no hay config, nada - DEBUG 
        return


    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        cursor = conn.cursor()

        query = """
        INSERT INTO logs (cylinders, displacement,horsepower,weight,acceleration,model_year,origin,mpg_pred)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        cylinders = input_data.get("cylinders")
        displacement  = input_data.get("displacement")
        horsepower  = input_data.get("horsepower")
        weight  = input_data.get("weight")
        acceleration  = input_data.get("acceleration")
        model_year  = input_data.get("model_year")
        origin  = input_data.get("origin")

        cursor.execute(query,
                        (cylinders, displacement,horsepower,weight,acceleration,model_year,origin,prediction)
                        )
        conn.commit()
        cursor.close()
        conn.close()

    except Error as e:
        print(f"Error al insertar en DB: {e}")



def preprocess_request(req: CarFeatures) -> pd.DataFrame:
    """
    Transforma la request a un DataFrame con las mismas columnas
    que se usaron para entrenar (model_columns).
    """
    base = {
        "cylinders": req.cylinders,
        "displacement": req.displacement,
        "horsepower": req.horsepower,
        "weight": req.weight,
        "acceleration": req.acceleration,
        "model_year": req.model_year,
    }

    # Manejo de origin
    # o origin_USA, origin_Europe, origin_Japan, etc.

    row = pd.DataFrame([base])

    # Creamos un DataFrame para origin y hacemos get_dummies
    origin_df = pd.DataFrame([{"origin": req.origin}])
    origin_dummies = pd.get_dummies(origin_df["origin"], prefix="origin")

    # Hacemos el join
    row = row.join(origin_dummies)

    # Asegurar que tiene exactamente las columnas de entrenamiento
    for col in model_columns:
        if col not in row.columns:
            row[col] = 0  # columna dummy faltante

    row = row[model_columns]

    return row






