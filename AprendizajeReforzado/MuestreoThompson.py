import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


datos = pd.read_csv("./Ads_CTR_Optimisation.csv")
N = 10000 # Numero de rondas  en las que se desea mostrar el anuncio
d = 10 # Numero de anuncion distintos a mostrar
anuncio_sele = []

Numero_ganancias = np.zeros(d,dtype = int)
Numero_perdidas = np.zeros(d,dtype = int)

ganancia_total = 0

for n in range(0, N):
    ad = 0
    max_parametro = 0
    for i in range(0, d):
        N1 = random.betavariate(Numero_ganancias[i]+1,Numero_perdidas[i]+1)
        if N1 > max_parametro:
            max_parametro = N1
            ad = i
    anuncio_sele.append(ad)
    ganancia = datos.values[n, ad]
    if ganancia == 1:
        Numero_ganancias[ad] = Numero_ganancias[ad] + 1
    else:
        Numero_perdidas[ad] = Numero_perdidas[ad] + 1
    ganancia_total = ganancia_total + ganancia
# Visualising the results
print(ganancia_total)
plt.hist(anuncio_sele)
plt.title('Histograma de selección de anuncios(Muestreo Thompson)')
plt.xlabel('Anuncio')
plt.ylabel('Número de veces en que el anuncio fue seleccionado')
plt.show()