#!/usr/bin/env python
# coding: utf-8

# In[93]:


import pandas as pd
import numpy as np
import random as rnd
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC

import os
os.environ['PATH'] += os.pathsep + '/Users/maria/opt/anaconda3/lib/python3.7/site-packages (0.13.2)'
style = "<style>svg{width:50% !important;height:50% !important;}</style>"
HTML(style)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold

get_ipython().run_line_magic('matplotlib', 'inline')


# In[176]:


train = pd.read_csv('C:\\Users\\anast\\OneDrive\\Рабочий стол\\train.csv')
test = pd.read_csv('C:\\Users\\anast\\OneDrive\\Рабочий стол\\test.csv')
gen_sub= pd.read_csv('C:\\Users\\anast\\OneDrive\\Рабочий стол\\gender_submission (1).csv')


# In[129]:


train.info()


# In[130]:


def missing_data(dataset, title):
    fig, ax = plt.subplots(figsize=(5,5))
    plt.title(title)  
    sns.heatmap(dataset.isnull(), cbar=False)


# In[131]:


missing_data(train, "Training Dataset")


# In[132]:


missing_data(test, "Test Dataset")


# In[133]:


train['Sex'].value_counts().to_frame()


# In[134]:


train.groupby('Sex').agg({'Survived':'mean'})


# In[135]:


train.groupby('Pclass').agg({'Survived':'mean'})


# In[136]:


pd.pivot_table(train, index = 'Survived', columns = 'Pclass', values = 'Ticket' ,aggfunc ='count')


# In[137]:


pd.pivot_table(train, index = 'Survived', columns = ['Pclass', "Sex"], values = 'Ticket' ,aggfunc ='count')


# In[138]:


train.groupby(['Pclass', "Sex"]).agg({'Survived':'mean'})


# In[139]:


visual=sns.FacetGrid(train,col='Survived')
visual.map(plt.hist,'Age',bins=10,color='green')


# In[140]:


visual=sns.FacetGrid(train,col='Survived',row='Pclass')
visual.map(plt.hist,'Age',bins=10,color='brown')
visual.add_legend()


# In[141]:


visual=sns.FacetGrid(train,col='Survived',row='Pclass')
visual.map(sns.scatterplot,'Age','Sex',color='green')
visual.add_legend()


# In[177]:


both=[train, test]


# In[178]:


for data in both:
    data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train.head()


# In[179]:


for data in both:
   data['Embarked'] = data['Embarked'].fillna('S')
   
train[['Embarked', 'Survived']].groupby(['Embarked']).mean().sort_values(by='Survived', ascending=False)


# In[180]:


for data in both:
    data['Embarked'] = data['Embarked'].map( {'S': 0, 'Q': 1, 'C': 2} ).astype(int)

train.head()


# In[181]:


train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Parch', 'SibSp'], axis=1)
test = test.drop(['Ticket', 'Cabin', 'Name', 'Parch', 'SibSp'], axis=1)


# In[206]:


X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop(['PassengerId', 'Age'],axis=1).copy()


# In[207]:


test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
test.head()


# In[208]:


train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand']).mean().sort_values(by='FareBand', ascending=True)


# In[209]:


for data in both:
    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare']

train = train.drop(['FareBand'], axis=1)
both = [train, test]
    
print(train.head())
print(test.head())


# In[210]:


X_train


# Logistic Regression

# In[211]:


LR = LogisticRegression()
LR.fit(X_train, Y_train)
Y_pred = LR.predict(X_test)
acc_LR = round(LR.score(X_train, Y_train) * 100, 2)
acc_LR


# KNN

# In[213]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# NAIVE BAYSE

# In[214]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# LINEAR SVC

# In[216]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# DECISION TREE

# In[218]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# RANDOM FOREST

# In[219]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[224]:


models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_knn, acc_LR, 
              acc_random_forest, acc_gaussian, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[228]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission


# In[229]:


submission.to_csv("submission1.csv",index=False)


# In[ ]:




