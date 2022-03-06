import time
from typing import Any, Union
import numpy as np
import pandas as pd
# import seaborn as sns
#import matplotlib.pyplot as plt
# from skimage.metrics import mean_squared_error
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
data = pd.read_csv('Movies_training_classification.csv')
data['Rotten Tomatoes'] = pd.to_numeric(data['Rotten Tomatoes'].str.strip('%')).div(100)
data.drop(['Age', 'Type'], axis=1, inplace=True)
data.dropna(axis=0, how='any', inplace=True)
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
#split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size =0.2, shuffle=False)
#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Logistic Regression
classifier = LogisticRegression(C=1.0)

Start_training_time1 = perf_counter()
classifier.fit(X_train, y_train)

# save the logistc model to disk
filename1 = 'logistic.sav'
pickle.dump(classifier, open(filename1, 'wb'))
End_training_time1 = perf_counter()

Start_test_time1 = perf_counter()
classifier.predict(X_test)
End_test_time1 =perf_counter()

acc_logistic =(classifier.score(X_test, y_test))
training_time_logistic = ( End_training_time1 - Start_training_time1)
testing_time_logistic  =(End_test_time1 -Start_test_time1)


#svm
svm = SVC()

Start_training_time2 = perf_counter()
svm.fit(X_train, y_train)
# save the svm model to disk
filename2 = 'svm.sav'
pickle.dump(svm, open(filename2, 'wb'))
End_training_time2 = perf_counter()

Start_test_time2 = perf_counter()
svm.predict(X_test)
End_test_time2 =perf_counter()

acc_svm = (svm.score(X_test, y_test))
training_time_SVM = ( End_training_time2 - Start_training_time2)
testing_time_SVM  =(End_test_time2 -Start_test_time2)

#Fitting knn  to the Training set
knn = KNeighborsClassifier(n_neighbors=23)
Start_training_time3 = perf_counter()
knn.fit(X_train, y_train)
# save the knn model to disk
filename3 = 'knn.sav'
pickle.dump(knn, open(filename3, 'wb'))
End_training_time3 = perf_counter()

Start_test_time3 = perf_counter()
knn.predict(X_test)
End_test_time3 =perf_counter()

acc_knn =(knn.score(X_test, y_test))
training_time_knn = ( End_training_time3 - Start_training_time3)
testing_time_knn  =(End_test_time3 -Start_test_time3)

#print_accuracy
print('Accuracy of Logistic regression classifier ', acc_logistic)
print('Accuracy of SVM classifier'  , acc_svm)
print('Accuracy of K-NN classifier'  , acc_knn)

#graph Accuracy
bar_data1 = {'logistic accurecy': acc_logistic, 'SVM accurecy': acc_svm, 'KNN accurecy': acc_knn}
models = list(bar_data1.keys())
values = list(bar_data1.values())
fig1 = plt.figure(figsize = (6, 3))
plt.bar(models, values, color =['maroon' ,'green' ,'blue' ], width = 0.4)
plt.xlabel("Models")
plt.ylabel("Accurecy")
plt.title("Accurecy for each model")
plt.show()

#graph training_time
bar_data2 = {'logistic_Train_time':training_time_logistic ,'Svm_Train_time': training_time_SVM, 'knn_Train_time': training_time_knn}
models = list(bar_data2.keys())
values = list(bar_data2.values())
fig2 = plt.figure(figsize = (6, 3))
plt.bar(models, values, color =['maroon' ,'green' ,'blue' ], width = 0.4)
plt.xlabel("Models")
plt.ylabel("Training time")
plt.title("Training time for each model")
plt.show()

#graph testing_time
bar_data3 = {'logistic_Test_Time': testing_time_logistic,'Svm_Test_Time': testing_time_SVM, 'Knn_Test_Time': testing_time_knn}
models = list(bar_data3.keys())
values = list(bar_data3.values())
fig3 = plt.figure(figsize = (6, 3))
plt.bar(models, values, color =['maroon' ,'green' ,'blue' ], width = 0.4)
plt.xlabel("Models")
plt.ylabel("Testing time")
plt.title("Testing time for each model")
plt.show()




