import time
from typing import Any, Union
import numpy as np
import pandas as pd
# import seaborn as sns
#import matplotlib.pyplot as plt
# from skimage.metrics import mean_squared_error
from pandas import Series, DataFrame
from pandas.io.parsers import TextFileReader
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from collections import Counter
from sklearn.datasets import make_blobs
from matplotlib import pyplot
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.svm import SVC
from time import perf_counter
import time
import pickle
# load data
data = pd.read_csv('Movies_testing_classification.csv')
data['Rotten Tomatoes'] = pd.to_numeric(data['Rotten Tomatoes'].str.strip('%')).div(100)
data.drop(['Age', 'Type'], axis=1, inplace=True)
cols = ['Genres', 'rate', 'Language', 'Country']
for col in cols:
    data[col].fillna(method='ffill', inplace=True)
movies_data = data.iloc[:, :]
#data.to_csv(r'removed data.csv')
X = data.iloc[:, 1:11]  # Features
Y = data['rate']  # Label
cols = ('Directors', 'Country', 'Language', 'Genres')
#labelEncoder
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(X[c].values))
    X[c] = lbl.transform(list(X[c].values))

#fill empty cells
X = X.groupby(X.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))
#apply label encoder on rate column
le = preprocessing.LabelEncoder()
Y = le.fit_transform(data['rate'])


#Logistic Regression
classifier = LogisticRegression()
# load the model from disk
filename1 = 'logistic.sav'
loaded_model = pickle.load(open(filename1, 'rb'))
result1 = loaded_model.score(X, Y)
print('acc_logistic ' ,result1)
#classifier.predict(X_test)
#svm
svm = SVC()
filename2 = 'svm.sav'
loaded_model = pickle.load(open(filename2, 'rb'))
result2 = loaded_model.score(X, Y)
print('acc_svm' ,result2)
#svm.predict(X_test)
#knn
knn = KNeighborsClassifier(n_neighbors=25)
filename3 = 'knn.sav'
loaded_model = pickle.load(open(filename3, 'rb'))
result3 = loaded_model.score(X, Y)
print('acc_knn' ,result3)
#knn.predict(X_test)

