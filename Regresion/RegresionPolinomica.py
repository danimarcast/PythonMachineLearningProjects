import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datos = pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/Regresion/Position_Salaries.csv")

x = datos.iloc[:,1:2].values
x.reshape(-1,1)
print(x)
y = datos.iloc[:,2].values

#print(x)
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 1234, test_size = 0.2)

from sklearn.linear_model import LinearRegression
modelo1 = LinearRegression()
modelo1.fit(x,y)

def polinomialCaracteristicas(g, x):
    x_ext = np.zeros((len(x),g+1)).astype(float)
    for i in range (0,g+1):
        for j in range (0,len(x)):
            x_ext[j,i]= x[j, 0] ** i
    return x_ext

x_ext = polinomialCaracteristicas(4,x)

modelo2 = LinearRegression()
modelo2.fit(x_ext,y)

y_pred = modelo2.predict(x_ext)
print(y_pred)
plt.scatter(x,y,color="red")
plt.plot(x,y_pred,color="blue")
plt.title('Verdad o Engaño (Regresion Polinomica)')
plt.xlabel('Nivel Posicion')
plt.ylabel('Salario')
#plt.show()

x_test = np.array([6.5],ndmin=2)

print(x_test)
obs = polinomialCaracteristicas(4,x_test)

print(obs)
pred = modelo2.predict(obs)
print(modelo1.predict([[6.5]]))
print(pred)