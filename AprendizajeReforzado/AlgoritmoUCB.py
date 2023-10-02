import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


datos = pd.read_csv("./Ads_CTR_Optimisation.csv")

N = 10000 # Numero de rondas  en las que se desea mostrar el anuncio
d = 10 # Numero de anuncion distintos a mostrar
anuncio_sele = []
num_sele = np.zeros(10, dtype = int)
suma_ganancia = np.zeros(10,dtype = int) # Suma de ganancias para cada anuncio
ganancia_total = 0

for n in range(0, N):
    ad = 0
    cotaSupMax = 0
    for i in range(0, d):
        if (num_sele[i] > 0):
            ganancia_prom = suma_ganancia[i] / num_sele[i]
            delta_i = np.sqrt(3/2 * np.log(n + 1) / num_sele[i])
            cotaSup = ganancia_prom + delta_i
        else:
            cotaSup = 1e400
        if cotaSup > cotaSupMax:
            cotaSupMax = cotaSup
            ad = i
    anuncio_sele.append(ad)
    num_sele[ad] = num_sele[ad] + 1
    ganancia = datos.values[n, ad]
    suma_ganancia[ad] = suma_ganancia[ad] + ganancia
    ganancia_total = ganancia_total + ganancia

# Visualising the results
plt.hist(anuncio_sele)
plt.title('Histograma de seleccion de anuncios')
plt.xlabel('Anuncio')
plt.ylabel('Numero de veces en que el anuncio fue seleccionado')
plt.show()