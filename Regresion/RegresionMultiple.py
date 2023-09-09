import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


datos = pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/Regresion/Startups.csv")
#print(datos)

x = datos.iloc[:,0:4].values
y = datos.iloc[:,4].values

#Codificacion para la columna state
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
Labelencoder_x=LabelEncoder()
x[:,3] = Labelencoder_x.fit_transform(x[:,3])
onehotencoder = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X = np.array(onehotencoder.fit_transform(x))
print(X)
print("Working............................")
X = X[:,1:]
print(X)

#Division del conjunto de datos
print("Working............................")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)
print(X_train)

# Ajuste del modelo multiple
from sklearn.linear_model import LinearRegression
Modelo1 = LinearRegression()
Modelo1.fit(X_train,y_train)

print("Working............................")
y_pred = Modelo1.predict(X_test)
print(y_pred)
print(y_test)
