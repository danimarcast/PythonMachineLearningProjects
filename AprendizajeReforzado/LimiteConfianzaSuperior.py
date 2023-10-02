import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


datos = pd.read_csv("./Ads_CTR_Optimisation.csv")

import random as rd
rd.seed(1234)
N = 10000
d = 10
ads_selec = []
Ganancia_total = 0
for n in range (0,N):
    ad = rd.randrange(d)
    ads_selec.append(ad)
    ganancia = datos.values[n,ad]
    Ganancia_total = Ganancia_total + ganancia

#print(Ganancia_total)
print(ads_selec)

# Visualización
plt.hist(ads_selec,color = "red")
plt.title("Anuncios seleccionados")
plt.xlabel("# Anuncio")
plt.ylabel("# Vistas del anuncio")
#plt.show()

