# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:58:55 2020

@author: Novin
"""

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import   cross_val_score
import time
import pandas as pd

def modify(s):
    vec = s.replace('[','')
    vec = vec.replace(']','')
    vec = vec.split(',')
    vec = map(float , vec)
    vec = list(vec)
    return vec
f = open ('result.txt','a')
f.write('accuracy\tprecision\trecall\tf1_macro\tf1_micro\n')
filenames = ['outputtopicInfo' + str(i) + '.xlsx' for i in range(10,310,10)]

for item in filenames:
        values = [0,0,0,0,0]
        dfDocument = pd.read_excel(item)
        y =  dfDocument['label']
        data = [modify(w) for w in dfDocument['vector']]
        j=0
        for score in ["accuracy", "precision", "recall" ,'f1_macro' , 'f1_micro']:
             
            values[j] = cross_val_score(SVC(), data, y,scoring=score, cv=10).mean()
            j = j+1
        for w in values:
            f.write(str(w)) 
            f.write('\t')
        f.write('\n')
        
        

f.close()