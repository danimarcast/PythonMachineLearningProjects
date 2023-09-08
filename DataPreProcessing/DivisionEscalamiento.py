# Importacion de lIbrerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Carga Datos
Datos = pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/venv/Data.csv")
#print(Datos)

X = Datos.iloc[:, :-1].values
Y = Datos.iloc[:, 3].values
#print(X)
#print(Y)


# Division del conjunto de datos en prueba y entrenamiento
from sklearn.model_selection import train_test_split

X_Test, X_Train,  Y_Test,  Y_Train = train_test_split(X,Y , test_size = 0.2, random_state = 1234)
#print(X_Test)
print(X_Train)

# Escalamiento de los datos
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

Esc_X = StandardScaler()
X_Train[:, 3:] = Esc_X.fit_transform(X_Train[:, 3:])
X_Test[:, 3:] = Esc_X.transform(X_Test[:, 3:])
print(X_Train)
print(X_Test)
#print(X_Train)