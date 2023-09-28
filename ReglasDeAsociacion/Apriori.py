import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datos = pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/ReglasDeAsociacion/Market_Basket_Optimisation.csv",header = None)

transaccion = []
for i in range(0, 7501):
  transaccion.append([str(datos.values[i,j]) for j in range(0, 20)])
# Entrenando el apriori
from apyori import apriori
reglas = apriori(transaccion, min_support = 0.003,min_confidence = 0.2, min_lift = 3,min_length= 2)

#visualizacion
resultados = list(reglas)
print(resultados[3])
