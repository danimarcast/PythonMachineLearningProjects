import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
datos = pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/Regresion/Position_Salaries.csv")

x = datos.iloc[:,1:2].values
x.reshape(-1,1)
print(x)
y = datos.iloc[:,2].values

# Escalado de variables
from sklearn.preprocessing import StandardScaler
escaler_x = StandardScaler()
escaler_y = StandardScaler()
y = escaler_y.fit_transform(y.reshape(-1,1)).reshape(-1,1)
x = escaler_x.fit_transform(x)


# Creación de una función que facilite la implementación de una regresión polinomial de cualquier grado deseado.
def polinomialCaracteristicas(g, x):
    x_ext = np.zeros((len(x),g+1)).astype(float)
    for i in range (0,g+1):
        for j in range (0,len(x)):
            x_ext[j,i]= x[j, 0] ** i
    return x_ext

x_ext = polinomialCaracteristicas(4,x)


modelo1 = LinearRegression()
modelo1.fit(x_ext,y)


# Ajuste del modelo de SVR al conjunto de datos
from sklearn.svm import SVR

modelo2 = SVR(kernel = 'rbf')
modelo2.fit(x,y)


print(escaler_y.inverse_transform(modelo2.predict(escaler_x.transform([[6.5]])).reshape(-1,1)))

y_pred = modelo2.predict(x)

#print(y_pred)


# Visualización de las SVR
plt.scatter(x,y,color="red")
plt.plot(x,y_pred,color="blue")
plt.title('Verdad o Engaño (SVR)')
plt.xlabel('Nivel Posicion')
plt.ylabel('Salario')
plt.show()