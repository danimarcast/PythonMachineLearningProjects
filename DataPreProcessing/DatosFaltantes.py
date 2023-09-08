
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


# Tratamiento de NAN
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
#print(X)