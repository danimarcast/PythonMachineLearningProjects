import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


datos = pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/Regresion/Startups.csv")
#print(datos)

x = datos.iloc[:,0:4].values
y = datos.iloc[:,4].values

#Codificacion para la columna state
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
Labelencoder_x=LabelEncoder()
x[:,3] = Labelencoder_x.fit_transform(x[:,3])
onehotencoder = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X = np.array(onehotencoder.fit_transform(x))
print(X)
print("Working............................")
X = X[:,1:]
print(X)

#Division del conjunto de datos
print("Working............................")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)
print(X_train)

# Ajuste del modelo multiple
from sklearn.linear_model import LinearRegression
Modelo1 = LinearRegression()
Modelo1.fit(X_train,y_train)

print("Working............................")
y_pred = Modelo1.predict(X_test)
print(y_pred)
print(y_test)

# Ajuste del modelo multiple realizando una Eliminación hacia atrás o back elimination.
import statsmodels.api as sm
X = np.append(arr = np.ones((len(X[:,0]),1)).astype(int), values = X ,axis = 1)
print(X)
X_opt = X[:, [0,1,2,3,4,5]] .tolist()# almacenamiento de las regresoras optimas
SL = 0.05

Modelo1_OLS = sm.OLS(endog = y, exog = X.tolist()).fit()
Modelo2_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(Modelo2_OLS.summary())

X_opt = X[:, [0,1,3,4,5]] .tolist()# almacenamiento de las regresoras optimas quitando el regresor x2
Modelo2_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(Modelo2_OLS.summary())

X_opt = X[:, [0,3,4,5]] .tolist()# almacenamiento de las regresoras optimas quitando el regresor x1 y x2
Modelo2_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(Modelo2_OLS.summary())

X_opt = X[:, [0,3,4,5]] .tolist()# almacenamiento de las regresoras optimas quitando el regresor x1 y x2
Modelo2_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(Modelo2_OLS.summary())

X_opt = X[:, [0,3,5]] .tolist()# almacenamiento de las regresoras optimas quitando el regresor x1 y x2 y x4
Modelo2_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(Modelo2_OLS.summary())

X_opt = X[:, [0,3]] .tolist()# almacenamiento de las regresoras optimas quitando el regresor x1 y x2 y x4 y X5
Modelo2_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(Modelo2_OLS.summary())


#Automatizando la eliminación hacia atras
# "x" corresponde a las variables regresoras que quiero incluir en mi modelo inicial y de estas posteriormente ir elimando las que tenga un
# un p-valor menor al nivel de significancia "SL".

X_opt = X[:, [0,1,2,3,4,5]] # almacenamiento de las regresoras optimas
SL = 0.05

import statsmodels.api as sm
def ELimHaciaAtras(x,SL):
    numVars = len(x[0])
    for i in  range(0,numVars):
        regresor_OLS = sm.OLS(y,x.tolist()).fit()
        maxVar = max(regresor_OLS.pvalues).astype(float)
        if maxVar > SL:
            for j in range(0,numVars - i):
                if (regresor_OLS.pvalues[j] == maxVar):
                    x = np.delete(x,j,axis = 1)
    print(regresor_OLS.summary())
    return x



ELimHaciaAtras(X_opt,SL)


# Eliminación hacia atras usando p-valor y R-ajustado
import statsmodels.formula.api as sm
def backElim(y,x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x.tolist()).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x