# Arquitectura_VDR_2024
Proyectos de la clase de Arquitectura de Producto de Datos

## Objective
Este proyecto busca hacer el pronóstico del precio de una casa dadas ciertas características del inmueble. Este repositorio contiene el modelo para la predicción del precio de una casa. Se desea que sea completamente automatizado. Se escogió XGBoost por mostrar el mejor desempeño entre los modelos probados (regresión lineal, regresión polinomial, random forest).

## Data
Los datos utilizados son abiertos y se pueden encontrar en Kaggle:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Steps
- prep.py : Este script busca preparar los datos de entrenamiento
  - Carga los datos
  - Selecciona las columnas necesarias
  - Realiza transformaciones que mejoran el desempeño del modelo
- training.py : Este script entrena y selecciona el mejor modelo (XGBoost)
  - Entrega un modelo entrenado
- inference.py : Este script realiza predicciones para un nuevo set de datos
  - Carga los datos nuevos
  - Le aplica el modelo entrenado
  - Regresa las predicciones
 
## Repository structure
src: contiene las funciones necesarias para el proyecto
data: aquí se almacenan los datos, el modelo y las predicciones
notebooks: contiene el EDA y el ejercicio inicial del desarrollo del modelo

## System requirements
Todo el proyecto se desarrolla en Python. En específico, se incluye el archivo de requirements.txt
