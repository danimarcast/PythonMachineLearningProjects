# Carga de librería
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datos =  pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/Regresion/Salary_Data.csv")
x = datos.iloc[:,0].values
y = datos.iloc[:,1].values


#Division de datos de prueba y entrenamiento
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1234)
