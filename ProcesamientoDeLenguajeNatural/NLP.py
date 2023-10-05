# Procesamiento del lenguaje natural
import pandas as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
opiniones = pd.read_csv("/Users/danimathud/Documents/GitHub/PythonMachineLearningProjects/ProcesamientoDeLenguajeNatural/Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

# Limpieza del texto
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# review = re.sub("[^a-zA-Z]"," " ,opiniones["Review"][0])
# review = review.lower() # Cambiar todas las letras a minusculas
# review = review.split()

# ps = PorterStemmer()

# review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]

# review = " ".join(review)

Op =[]

for i in range (0,1000):
    review = re.sub("[^a-zA-Z]"," " ,opiniones["Review"][i])
    review = review.lower() # Cambiar todas las letras a minusculas
    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]

    review = " ".join(review)
    Op.append(review)
    
# Creación de la bolsa de palabras
from sklearn.feature_extraction.text import CountVectorizer

CV = CountVectorizer(max_features = 1500)

x = CV.fit_transform(Op).toarray()

y = opiniones.iloc[:,1].values

# Partición en datos de prueba y entrenamiento
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1, random_state = 1234)


# Escalado de variables
from sklearn.preprocessing import StandardScaler
escalador = StandardScaler()
x_train = escalador.fit_transform(x_train)
x_test = escalador.transform(x_test)

# Clasificando via Naive Bayes
from sklearn.naive_bayes import GaussianNB

clasificador = GaussianNB()
clasificador.fit(x_train,y_train)

y_pred = clasificador.predict(x_test)

ConfMat = confusion_matrix(y_test,y_pred)
print(ConfMat)





