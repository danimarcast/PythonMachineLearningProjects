import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datos = pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/Regresion/Position_Salaries.csv")

x = datos.iloc[:,1:2].values
y = datos.iloc[:,2:3].values

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 1234, test_size = 0.2)