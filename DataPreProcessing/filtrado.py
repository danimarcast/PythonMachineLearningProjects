import xlrd as xlrd
import pandas as pd

usuarios = pd.read_excel("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/DataPreProcessing/usuarios.xlsx",)

curp =  []

años_deseados =[93,94,95,96,97,98,99,0,1,2,3,4,5,6,7,8,9,10,11]

años = []
filtrados = []

print(usuarios.shape)
for i in range(0,2017):
    if type(usuarios["CURP"][i]) == int:
        usuarios.drop(i, axis = 0)
usuarios = usu

print(usuarios.shape)
usuarios.reset_index(drop = True, inplace = True)
print(usuarios.shape)

# for i in range(0,len(usuarios)):
#     for j in range(0,10):
#         if usuarios["CURP"][i][4] != str(j):
#             usuarios.drop(i, axis = 0)
            
            

# print(usuarios.shape)˙