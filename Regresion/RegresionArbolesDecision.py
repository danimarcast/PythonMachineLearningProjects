import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CART (Classification and regression trees)
datos = pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/Regresion/Position_Salaries.csv")

x = datos.iloc[:,1:2].values
x.reshape(-1,1)
y = datos.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
modelo1 = DecisionTreeRegressor(random_state = 1234)
modelo1.fit(x,y)

print(modelo1.predict([[6.5]]).reshape(-1,1))

#x_grid = np.arange(min(x),max(x),0.1).reshape(-1,1)
y_pred = modelo1.predict(x)
plt.scatter(x,y,color="red")
plt.plot(x,y_pred,color="blue")
plt.title('Verdad o Engaño (CART)')
plt.xlabel('Nivel Posicion')
plt.ylabel('Salario')
plt.show()