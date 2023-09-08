# Ajusta un modelo de regresión sin validar supuestos del modelo.


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

#Ajuste del modelo de regresion simple

from sklearn.linear_model import LinearRegression

x_train = x_train[:,None] # usar este formato para poner el array de variable regresora como vector columna, esto en caso de tener un solo regresor (regresion simple)
# como forma alternativa se puede usar la función reshape ex "x_train=x_train.reshape(-1,1)"
#print(x_train)
#print(y_train)
modelo1 = LinearRegression()
modelo1.fit(x_train,y_train)

# Prediciendo haciendo uso del modelo lineal
y_pred = modelo1.predict(x_test.reshape(-1,1))
print(y_pred)
print(y_test)


# Dispersion de los datos
plt.scatter(x_train.reshape(-1,1),y_train,color="red")
plt.plot(x_train.reshape(-1,1),modelo1.predict(x_train.reshape(-1,1)),color="blue")
plt.title("Sueldo vs Años de experiencia (Datos de prueba)")
plt.ylabel("Salario(dolares)")
plt.xlabel("Años de experiencia")
plt.show()
plt.savefig("SalarioVsExperiencia")