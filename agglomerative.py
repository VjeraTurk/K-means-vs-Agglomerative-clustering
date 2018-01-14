#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 18:43:24 2018

@author: vjera
"""
'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''
import pandas as pd

fname = '/home/vjera/Desktop/Link to STROJNO/Projekt/titanic/train.csv' #train!!
fname2 = '/home/vjera/Desktop/Link to STROJNO/Projekt/titanic/test.csv' #test

data = pd.read_csv(fname)
data_test = pd.read_csv(fname2)
#print(len(data))
#print(data.head())
#data.count()

data_num = data
data_num.drop(['PassengerId', 'Name'], 1, inplace=True)
data_num.convert_objects(convert_numeric=True)
data_num.fillna(0, inplace=True)

data_test_num = data_test
#data_test_num.drop(['PassengerId', 'Name'], 1, inplace=True)
#data_test_num.convert_objects(convert_numeric=True)
#data_test_num.fillna(0, inplace=True)

############################################################
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from time import time
############################################################
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

df = pd.read_excel('titanic.xls')
#print(df.head())
df.drop(['body','name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
#print(df.head())

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
data_num=handle_non_numerical_data(data_num)


X = np.array(df.drop(['survived'], 1).astype(float))
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct/len(X))



X = np.array(data_num.drop(['Survived'], 1).astype(float))
#X.shape (891, 10)
y = np.array(data_num['Survived'])

agg = AgglomerativeClustering(n_clusters=2).fit(X);

for linkage in ( 'average', 'complete'):
    for affinity in ( 'euclidean', 'l1', 'l2', 'manhattan', 'cosine'):
        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=2,affinity=affinity)
        t0 = time()
        clustering.fit(X)
        correct = 0
        wrong = 0
        for ind,(i,j) in enumerate(zip(y,clustering.labels_)):
            if i == j :
                correct += 1        
            else :
                wrong +=1
        print(correct/len(y)*100,"%")
        print("%s : %.2fs" % (linkage, time() - t0))



clustering = AgglomerativeClustering(linkage='ward', n_clusters=2,affinity='euclidean')
t0 = time()
clustering.fit(X)
correct = 0
wrong = 0
for ind,(i,j) in enumerate(zip(y,clustering.labels_)):
    if i == j :
        correct += 1        
    else :
        wrong +=1
print(correct/len(y)*100,"%")
print("%s : %.2fs" % (linkage, time() - t0))


