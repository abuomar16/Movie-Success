import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from time import perf_counter
import pickle

#Load data
data = pd.read_csv('Movies_testing.csv')
data['Rotten Tomatoes'] = pd.to_numeric(data['Rotten Tomatoes'].str.strip('%')).div(100)
# any=1 cell , all= total cell
#data.dropna(axis = 0 ,how='any', inplace=True)
data['IMDb']=data['IMDb'].fillna(data['IMDb'].mean())
data['Runtime']=data['Runtime'].fillna(data['Runtime'].mean())
data.drop(['Age' ,'Type'],axis=1, inplace=True)

cols = ['Genres' ,'Directors', 'Language', 'Country']
for col in cols:
    data[col].fillna(method='ffill', inplace=True)

movies_data=data.iloc[:,:]
X = data.iloc[:,1:11] #Features
Y = data['IMDb'] #Label
cols = ('Directors','Country','Language')
#convert string to num
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(X[c].values))
    X[c] = lbl.transform(list(X[c].values))


# treat null values
X['Genres'].fillna(' ', inplace = True)
# separate all genres into one list, considering comma + space as separators
genre = X['Genres'].str.split(',')
# flatten the list
flat_genre = [item for sublist in genre for item in sublist]
# convert to a set to make unique
set_genre = set(flat_genre)
# back to list
unique_genre = list(set_genre)
# create columns by each unique genre
Z= X.reindex(X.columns.tolist() + unique_genre, axis=1, fill_value=0)
# for each value inside column, update the dummy
for index, row in Z.iterrows():
    for val in row.Genres.split(','):
        if val != ' ':
            Z.loc[index, val] = 1

Z.drop('Genres', axis = 1, inplace = True)
Z = Z.groupby(Z.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))


#load_model

filename2 = 'pol_feature.sav'
loaded_model2 = pickle.load(open(filename2, 'rb'))

filename1 = 'pol_model.sav'
loaded_model1 = pickle.load(open(filename1, 'rb'))


prediction = loaded_model1.predict(loaded_model2.transform(Z))
r2 = r2_score(Y, prediction)


print('Mean Square Error', metrics.mean_squared_error(Y, prediction))
print('r2 score for polynomial model is', r2)