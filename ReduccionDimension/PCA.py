import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# PCA para reducción de dimnensión

vino = pd.read_csv("./Wine.csv")

x = vino.iloc[:, 0:13].values
y = vino.iloc[:, 13].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

# escalado de variables

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Implementacion de PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = 2 ,random_state = 1234)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

varianza_expl = pca.explained_variance_ratio_

clasificador = LogisticRegression()
clasificador.fit(x_train,y_train)

y_pred = clasificador.predict(x_test)


# Contrastar que tan bien clasifico la regresion logistica

from sklearn.metrics import confusion_matrix

MatrizcConfusion = confusion_matrix(y_test,y_pred)
print(MatrizcConfusion)

# Visualizacion de la clasificacion
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clasificador.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green',"blue")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green',"blue"))(i), label = j)
plt.title('Regresion Logistica(Datos Entrenamiento)')
plt.xlabel('CP1')
plt.ylabel('CP2')
plt.legend()
plt.show()








