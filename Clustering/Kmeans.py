import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
#Preprocesado.......................................
#Carga del conjunto de datos
datos = pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/Clustering/Mall_Customers.csv")
x = datos.iloc[:,3:5].values


#Metodo del codo para selección del número adecuado de clusters
Distancias = []

for i in range(1,11):
    k_means = KMeans(n_clusters = i, init = "k-means++", max_iter = 100, n_init = 10, random_state = 1234)
    k_means.fit(x)
    Distancias.append(k_means.inertia_)
plt.subplot(211)
plt.plot(range(1,11),Distancias)
plt.xlabel("Numero de clusters")
plt.ylabel("Distancia al cluster ")
plt.title("Grafico de codo ")

# Del grafico del codo podemos observar que es adecuado elegir un total de 5 clusters

k_means1 = KMeans(n_clusters = 5, init = "k-means++", max_iter = 100, random_state = 1234, n_init = 10)
y_pred = k_means1.fit_predict(x)
print(y_pred)

# Visualización de la segmentación de los datos
plt.subplot(212)
plt.scatter(x[y_pred==0,0],x[y_pred==0,1], c = "b", label = "Cluster 1")
plt.scatter(x[y_pred==1,0],x[y_pred==1,1], c = "g", label = "Cluster 2")
plt.scatter(x[y_pred==2,0],x[y_pred==2,1], c = "r",label = "Cluster 3")
plt.scatter(x[y_pred==3,0],x[y_pred==3,1], c = "c",label = "Cluster 4")
plt.scatter(x[y_pred==4,0],x[y_pred==4,1], c = "m",label = "Cluster 5")
plt.scatter(k_means1.cluster_centers_[:,0],k_means1.cluster_centers_[:,1], c = "k", label = "Centros")
plt.xlabel("Ingreso Anual (dolares)")
plt.ylabel("Puntuacion Gastos")
plt.legend()
plt.show()

