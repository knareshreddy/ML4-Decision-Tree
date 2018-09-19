#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 21:52:38 2018
@author: naresh

"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

df=pd.read_csv('https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv')
#df=pd.read_excel('Titanic.xlsx')

df.columns.tolist()
df=df.reindex_axis(['PassengerId',
 'Survived',
 'Name',
 'Pclass',
 'Sex',
 'Age',
 'SibSp',
 'Parch',
 'Fare',
 'Ticket',
 'Cabin',
 'Embarked'], axis=1)


#Check to see any null values 

df.isnull().sum()
df['Age'].isnull().sum()

#Dropping Null Values of a Variable
#df.dropna( subset=['Age'], inplace=True)


# or Imputing Missing Values 
 imputer=Imputer(missing_values="NaN", strategy="mean", axis=0)
df['Age']=imputer.fit_transform(df[['Age']])
#or 
#df['Age'].fillna(df['Age'].mean(), inplace=True)


#Pclass, Sex, Age, SibSp (Siblings aboard), Parch (Parents/children aboard) and Fare
X=pd.DataFrame( df.iloc[:, 3:9].values)
y=df.iloc[:, 1 ]



#Convert the Categorical Variables to numeric with Label encoder and Onehot encoder

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
X.iloc[:, 1] = labelencoder_x_1.fit_transform(X.iloc[:, 1])
onehotencoder_1 = OneHotEncoder(categorical_features = [1])
X = onehotencoder_1.fit_transform(X).toarray()


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)



# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Model Preparation
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy', max_depth=5)
classifier.fit(X_train, y_train)

#Fit the model
y_pred = classifier.predict(X_test)

#Find the accuracy of Model

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm=confusion_matrix(y_test, y_pred)

accuracy=accuracy_score(y_test,y_pred)


# ********Model gives ~ 83% accuracy*******

#print(classification_report(y_test,y_pred))


from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator=classifier, X=X, y=y, scoring='accuracy', cv=10)
scores, scores.mean()

#*******With Cross Validation Model mean score was obtained at ~82%*********
































































































