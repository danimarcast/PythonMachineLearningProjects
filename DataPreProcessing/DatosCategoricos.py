#Codificacion Variable independiente
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Importacion de lIbrerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Carga Datos
Datos = pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/venv/Data.csv")
#print(Datos)

X = Datos.iloc[:, :-1].values
Y = Datos.iloc[:, 3].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#Codificando Variable dependiente
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)