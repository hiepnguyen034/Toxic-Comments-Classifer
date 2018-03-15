
import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data=data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],1)
data=data.rename(columns={"v1":"Classification","v2":"SMS"})
data=data[['SMS','Classification']]
data.describe()
data['Classification']=data['Classification'].map({'spam':1,'ham':0})

    
    
    
x_train,x_test,y_train,y_test=train_test_split(data['SMS'],data['Classification'],
                                                   test_size=0.15, random_state = 5)

tv=TfidfVectorizer(stop_words="english")

tv.fit(x_train)
x_train_df=tv.transform(x_train)
x_test_df=tv.transform(x_test)
prediction = dict()

#Naive Bayes Classification

#Multinomial Naive Bayes
mb=MultinomialNB()
y_pred=mb.fit(x_train_df,y_train)
prediction["MBayes"]=y_pred.predict(x_test_df)
accuracy_score(y_test,prediction["MBayes"])
#Accuracy score is 0.972

#Gaussian Naive Bayes
gnb = GaussianNB()
x_train_df=x_train_df.toarray()
x_test_df=x_test_df.toarray()
y_pred=gnb.fit(x_train_df,y_train)

prediction["GBayes"]=y_pred.predict(x_test_df)
accuracy_score(y_test,prediction["GBayes"])
#Accuracy score is 0.887

#Support Vecctor Machine

x_train,x_test,y_train,y_test=train_test_split(data['SMS'],data['Classification'],
                                                   test_size=0.15, 
                                                   random_state = 5)
clf=svm.SVC()
clf.fit(x_train_df,y_train)
y_pred=clf.fit(x_train_df,y_train)
prediction["SVM"]=y_pred.predict(x_test_df)
accuracy_score(y_test,prediction["SVM"])

#Accuracy score is 0.883

#Logistic Regression

clf=LogisticRegression()
clf.fit(x_train_df,y_train)
y_pred=clf.fit(x_train_df,y_train)
prediction["Logistic"]=y_pred.predict(x_test_df)
accuracy_score(y_test,prediction["Logistic"])
#Accuracy score is 0.971

