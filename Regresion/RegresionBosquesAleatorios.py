import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datos = pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/Regresion/Position_Salaries.csv")

x = datos.iloc[:,1:2].values
x.reshape(-1,1)
y = datos.iloc[:,2].values

# Regresión con Bosques aleatorios
from sklearn.ensemble import RandomForestRegressor

modelo1 = RandomForestRegressor(n_estimators = 10, criterion = "squared_error",random_state = 1234)
modelo1.fit(x,y)

print(modelo1.predict([[6.5]]).reshape(-1,1))

x_grid = np.arange(min(x),max(x),0.1).reshape(-1,1)
y_pred = modelo1.predict(x_grid)


plt.scatter(x,y,color="red")
plt.plot(x_grid,y_pred,color="blue")
plt.title('Verdad o Engaño (CART)')
plt.xlabel('Nivel Posicion')
plt.ylabel('Salario')
plt.show()