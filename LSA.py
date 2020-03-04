# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 05:39:07 2020

@author: Novin
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import time 
from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer



df = pd.read_excel('Cf output.xlsx')


y = df['label60']
data = []
for rowIndex,row in df.iterrows():
    data.append(str(row['title']) + str(row['articleBody']))

X_train_raw, X_test_raw, y_train, y_test = train_test_split(data, y, test_size=0.2)    
    
vectorizer = TfidfVectorizer(max_df=0.5, max_features=15000,
                             min_df=3, stop_words='english',
                             use_idf=False)

# Build the tfidf vectorizer from the training data ("fit"), and apply it 
# ("transform").
X_train_tfidf = vectorizer.fit_transform(X_train_raw)
svd = TruncatedSVD(210)
lsa = make_pipeline(svd, Normalizer(copy=False))

X_train_lsa = lsa.fit_transform(X_train_tfidf)

explained_variance = svd.explained_variance_ratio_.sum()
print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))


# Now apply the transformations to the test data as well.
X_test_tfidf = vectorizer.transform(X_test_raw)
X_test_lsa = lsa.transform(X_test_tfidf)

# ################# model #################################################
clf = svm.SVC(random_state=42 , gamma = 'auto')
print("Default Parameters are: \n",clf.get_params)

start_time = time.time()
clf.fit(X_train_lsa, y_train)
fittime = time.time() - start_time
print("Time consumed to fit model: ",time.strftime("%H:%M:%S", time.gmtime(fittime)))
start_time = time.time()
score=clf.score(X_test_lsa, y_test)
print("Accuracy: ",score)

y_pred = clf.predict(X_test_lsa)

scoretime = time.time() - start_time
print("Time consumed to score: ",time.strftime("%H:%M:%S", time.gmtime(scoretime)))
case1=[score,fittime,scoretime]
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
