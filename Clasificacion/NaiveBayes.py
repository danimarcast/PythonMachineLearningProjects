import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC

#Preprocesado.......................................
#Carga del conjunto de datos
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

# Implementacion del algoritmo.........................
from sklearn.naive_bayes import GaussianNB

clasificador = GaussianNB()
clasificador.fit(x_train,y_train)

y_pred = clasificador.predict(x_test)

ConfMat = confusion_matrix(y_test,y_pred)
print(ConfMat)

# Visualizacion de la clasificacion
X_set, y_set = escalador.inverse_transform(x_train), y_train


X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 100))
plt.contourf(X1, X2, clasificador.predict(escalador.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('#008B8B', 'darkviolet')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('Green', 'fuchsia'))(i), label = j)
plt.title('Naive Bayes (Datos Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Salario')
plt.legend()
plt.show()
