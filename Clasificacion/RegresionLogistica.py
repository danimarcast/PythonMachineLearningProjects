
# Regresión Logística

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datos = pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/Clasificacion/Social_Network_Ads.csv")

x = datos.iloc[:,0:2]
y = datos.iloc[:,2].values


# Partición en datos de prueba y entrenamiento
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 1234)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
escalador = StandardScaler()
x_train = escalador.fit_transform(x_train)
x_test = escalador.transform(x_test)



# Ajustar del modelo de regresión logística

from sklearn.linear_model import LogisticRegression

clasificador = LogisticRegression()
clasificador.fit(x_train,y_train)

y_pred = clasificador.predict(x_test)


# Contrastar que tan bien clasifico la regresion logistica
from sklearn.metrics import confusion_matrix

MatrizcConfusion = confusion_matrix(y_test,y_pred)
print(MatrizcConfusion)

# Visualizacion de la clasificacion
from matplotlib.colors import ListedColormap
X_set, y_set = escalador.inverse_transform(x_train)
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, clasificador.predict(escalador.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Regresion Logistica(Datos Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Salario')
plt.legend()
plt.show()