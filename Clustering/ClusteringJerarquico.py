import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
#Preprocesado.......................................
#Carga del conjunto de datos
datos = pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/Clustering/Mall_Customers.csv")
x = datos.iloc[:,3:5].values

# Encontrar el numero optimo de clusters usando el dendograma
plt.subplot(211)
dendograma = sch.dendrogram(sch.linkage(x,method = "ward",metric = "euclidean"))
plt.title("Dendograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia ")
#plt.show()

# Ajuste del clustering jerárquico
from sklearn.cluster import AgglomerativeClustering
ClusJerar = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
ClusJerar.fit(x)
y_clus = ClusJerar.fit_predict(x)
plt.subplot(212)
plt.scatter(x[y_clus==0,0],x[y_clus==0,1], c = "b", label = "Cluster 1")
plt.scatter(x[y_clus==1,0],x[y_clus==1,1], c = "g", label = "Cluster 2")
plt.scatter(x[y_clus==2,0],x[y_clus==2,1], c = "r",label = "Cluster 3")
plt.scatter(x[y_clus==3,0],x[y_clus==3,1], c = "c",label = "Cluster 4")
plt.scatter(x[y_clus==4,0],x[y_clus==4,1], c = "m",label = "Cluster 5")
plt.xlabel("Ingreso Anual (dolares)")
plt.ylabel("Puntuacion Gastos")
plt.legend()
plt.show()