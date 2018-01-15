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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from time import time
############################################################
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
############################################################
fname_train = '/home/vjera/Desktop/Link to STROJNO/Projekt/titanic/train.csv' #train
fname_test = '/home/vjera/Desktop/Link to STROJNO/Projekt/titanic/test.csv' #test

def agglomerative(linkage,affinity):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters,affinity=affinity)
    t0 = time()
    clustering.fit(X)
    print("%s : %.2fs" % (linkage, time() - t0))
    print(linkage, affinity)
    correct = 0
    wrong = 0
    
    for ind,(i,j) in enumerate(zip(y,clustering.labels_)):
        if i == j :
            correct += 1        
        else :
            wrong +=1
    if (correct/len(y)) > 0.5:
        print(correct/len(y)*100,"%\n")
    else :
        print((1-correct/len(y))*100,"% !!!\n")    
    return clustering

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

############# to nomeric data
data = pd.read_csv(fname_train) #train
data_num = pd.DataFrame.copy(data) #zelimo zadržati originalne podatke
data_num.drop(['PassengerId', 'Name'], 1, inplace = True) #ime i id ne utječu na rezultat
data_num.fillna(0, inplace=True)
data_num = handle_non_numerical_data(data_num)

#data_num.drop(['Cabin'], 1, inplace=True)



from sklearn.decomposition import PCA
variance_pct = .99
# Create PCA object
pca = PCA(n_components=variance_pct)
# Transform the initial features
X_transformed = pca.fit_transform(data_num,data_num['Survived'])
# Create a data frame from the PCA'd data
pcaDataFrame = pd.DataFrame(X_transformed)
print(pcaDataFrame.shape[1], " components describe ", str(variance_pct)[1:], "% of the variance")


############ preprocesing
X = np.array(data_num.drop(['Survived'], 1).astype(float))
y = np.array(data_num['Survived'])
X = preprocessing.scale(X)









n_clusters=2
#####################   K-means   #######################
print("K-means")
clf = KMeans(n_clusters=n_clusters, n_init = 100)
clf.fit(X)
clf.cluster_centers_
correct = 0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
if (correct/len(y)) > 0.5:
    print(correct/len(y)*100,"%\n")
else :
    print((1-correct/len(y))*100,"%\n")

##############   AgglomerativeClustering   ##################
#X.shape (891, 10)
for linkage in ( 'average', 'complete'):
    for affinity in ( 'euclidean', 'l1', 'l2', 'manhattan', 'cosine'):
        clusterin=agglomerative(linkage,affinity)
###########   AgglomerativeClustering: WARD ##################
#best after scaling
#0.6742 on test
clustering= agglomerative('ward','euclidean')

###########   AgglomerativeClustering: complete cosine #######
#best after scaling+drop Cabin
#clustering= agglomerative('complete','cosine')
#worse on test data!
d = { 'predicted': clusterin.labels_,'Survived': data.Survived}
result_df= pd.DataFrame(d)

#######################   testing ############################
data = pd.read_csv(fname_test)
data_num = pd.DataFrame.copy(data) #zelimo zadržati originalne podatke
data_num.drop(['PassengerId', 'Name'], 1, inplace = True) #ime i id ne utječu na rezultat
data_num.fillna(0, inplace=True)
data_num = handle_non_numerical_data(data_num)

X = preprocessing.scale(X)
X_test = np.array(data_num).astype(float)
X_test = preprocessing.scale(X_test)

prediction = clustering.fit_predict(X_test)
data.PassengerId
d = { 'PassengerId': data.PassengerId,'Survived': prediction}
result=pd.DataFrame(d)
#result.to_csv(path_or_buf='/home/vjera/Desktop/Link to STROJNO/Projekt/titanic/result2.csv', index=False)






#cijela baza titanic.xls
df = pd.read_excel('titanic.xls')
df.drop(['body','name'], 1, inplace=True)
#df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
df = handle_non_numerical_data(df)

