# Plantilla de Pre Procesado....

# Importacion de lIbrerias


# Carga Datos
Datos = pandas.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/venv/Data.csv")
#print(Datos)

X = Datos.iloc[:, :-1].values
Y = Datos.iloc[:, 3].values
#print(X)
#print(Y)


# Tratamiento de NAN
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
#print(X)

#Codificacion Variable independiente
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
#print(X)

#Codificando Variable dependiente
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Y = le.fit_transform(Y)
#print(Y)

# Division del conjunto de datos en prueba y entrenamiento
from sklearn.model_selection import train_test_split

X_Test, X_Train,  Y_Test,  Y_Train = train_test_split(X,Y , test_size = 0.2, random_state = 1234)
#print(X_Test)
#print(X_Train)

# Escalamiento de los datos
from sklearn.preprocessing import StandardScaler

Esc_X = StandardScaler()
X_Train = Esc_X.fit_transform(X_Train)
X_Test = Esc_X.transform(X_Test)