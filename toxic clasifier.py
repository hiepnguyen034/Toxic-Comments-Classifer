import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv('train.csv').fillna('')


def clf_model(data,types,test):
    prediction=dict()
    x_train,x_test,y_train,y_test=train_test_split(data['comment_text'],data[types],
                                                   test_size=0.3, random_state = 5)
    tv=TfidfVectorizer(stop_words="english")
    tv.fit(x_train)
    train_df=tv.transform(x_train)
    test_df=tv.transform(x_test)
    if test == 'logistic':
        clf=LogisticRegression()
        clf.fit(train_df,y_train)
        y_pred=clf.fit(train_df,y_train)
        prediction["Logistic"]=y_pred.predict(test_df)
        return accuracy_score(y_test,prediction["Logistic"])
    elif test == 'tree':
        clf=GradientBoostingClassifier()
        clf.fit(train_df,y_train)
        y_pred=clf.fit(train_df,y_train)
        prediction["Trees"]=y_pred.predict(test_df)
        return accuracy_score(y_test,prediction["Trees"])
    elif test == 'mlp':
       clf=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15, ), random_state=1)
       clf.fit(train_df,y_train)
       y_pred=clf.fit(train_df,y_train)
       prediction["MLP"]=y_pred.predict(test_df)
       return accuracy_score(y_test,prediction["MLP"]) 

log_test=[]
types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for i in types:
    log_test.append(clf_model(data,i,'logistic'))

tree_test=[]
types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for i in types:
    tree_test.append(clf_model(data,i,'tree'))

neural_test=[]  
types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for i in types:
    neural_test.append(clf_model(data,i,'mlp'))

d={'Offense':types,'Logistic Accuracy':log_test, 'Trees Accuracy': tree_test,
   'Neural Net Accuracy': neural_test}
data= pd.DataFrame(d)

data=data[['Offense','Logistic Accuracy','Trees Accuracy','Neural Net Accuracy']]
data.to_csv('Accuracy')
    
