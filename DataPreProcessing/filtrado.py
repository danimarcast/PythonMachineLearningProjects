import xlrd as xlrd
import pandas as pd

usuarios = pd.read_excel("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/DataPreProcessing/usuarios.xlsx",)

curp =  []

años_deseados =[93,94,95,96,97,98,99,0,1,2,3,4,5,6,7,8,9,10,11]

años = []
filtrados= []

for i in range(0,2017):
    if type(usuarios["CURP"][i]) == int or len(usuarios["CURP"][i]) != 18:
        usuarios = usuarios.drop(i, axis = 0)
        
usuarios.reset_index(drop = True, inplace = True)

for j in range(0,len(usuarios)):
    for i in range(0,10):
        if usuarios["CURP"][j][4] != str(i):
           usuarios.drop(j, axis = 0)
           usuarios.reset_index(drop = True, inplace = True)

usuarios = usuarios.drop(532, axis = 0)
usuarios.reset_index(drop = True, inplace = True)

usuarios = usuarios.drop(594, axis = 0)
usuarios.reset_index(drop = True, inplace = True)

usuarios = usuarios.drop(751, axis = 0)
usuarios.reset_index(drop = True, inplace = True)

usuarios = usuarios.drop(1062, axis = 0)
usuarios.reset_index(drop = True, inplace = True)

for i in range(0,len(usuarios)):
    for j in range(0,len(años_deseados)):
        if int(usuarios["CURP"][i][4:6]) == años_deseados[j]:
            if usuarios["CURP"][i][11:13]== "GT":
                filtrados.append(usuarios["CURP"][i])
                

file = open('filtrados.txt','w')
for  usuario in filtrados:
	file.write(usuario +"\n")
file.close()
        