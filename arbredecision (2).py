# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score,confusion_matrix,plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

df=pd.read_excel('breast-w-excelNombre.xlsx')
df['Class'] = df['Class'].map({'malignant':1,'benign':0})

df.info()

df.columns

df.describe()

df.describe()
plt.hist(df['Class'])
plt.show()



g=df.drop('Class', axis=1, inplace=False)

X_train,X_test,Y_train,Y_test= train_test_split(df[g.columns],df['Class'],random_state=0, test_size=0.1)

model=DecisionTreeClassifier()

model.fit(X_train,Y_train)

fn=g.columns
cl=['malignant','benign']
fig,axes =plt.subplots(nrows=1,ncols=1,figsize=(30,30))
tree.plot_tree(model,feature_names=fn,class_names=cl,filled=True,fontsize=13);
fig.savefig('arbre.png')

ypred = model.predict(X_test)

diff =np.array([Y_test,ypred])
diff

predictions = model.predict(X_test[X_test.columns])
accuracy = accuracy_score(predictions,Y_test)
print(accuracy*100)

score = model.score(X_test,Y_test)
print(score*100)

ypred = model.predict(X_test)
confusion_matrix(Y_test,ypred)

plot_confusion_matrix(model,X_test,Y_test,display_labels=['benign','malignant'])

plt.figure(figsize=(16,10))
sns.heatmap(df.corr(),annot=True)
plt.show()

from sklearn.ensemble import ExtraTreesClassifier
X=df.iloc[:,0:9] 
y=df.iloc[:,-1]
model = ExtraTreesClassifier()
model.fit(X,y)
feat_importances = pd.Series(model.feature_importances_,index=X.columns)
feat_importances .nlargest(9).plot(kind='bar')
plt.show()

#modele selon la matrice

new=df.drop(['Class','Clump_Thickness','Marginal_Adhesion','Single_Epi_Cell_Size','Normal_Nucleoli','Mitoses'], axis=1, inplace=False)
train_X,test_X,train_Y,test_Y= train_test_split(df[new.columns],df['Class'],random_state=0, test_size=0.1)

newmodel=DecisionTreeClassifier(max_depth=4)
newmodel.fit(train_X,train_Y)
fn=new.columns
cl=['malignant','benign']
fig,axes =plt.subplots(nrows=1,ncols=1,figsize=(40,40))
tree.plot_tree(newmodel,feature_names=fn,class_names=cl,filled=True,fontsize=12);
fig.savefig('tree.png')

score = newmodel.score(test_X,test_Y)
print(score*100)

#confusion matrix
ypredit = newmodel.predict(test_X)
confusion_matrix(test_Y,ypredit)

plot_confusion_matrix(newmodel,test_X,test_Y,display_labels=['benign','malignant'])

#modele selon la fonction

new=df.drop(['Class','Clump_Thickness','Marginal_Adhesion','Single_Epi_Cell_Size','Bland_Chromatin','Mitoses'], axis=1, inplace=False)
train_X,test_X,train_Y,test_Y= train_test_split(df[new.columns],df['Class'],random_state=0, test_size=0.1)

newmodel=DecisionTreeClassifier(max_depth=4)
newmodel.fit(train_X,train_Y)
fn=new.columns
cl=['malignant','benign']
fig,axes =plt.subplots(nrows=1,ncols=1,figsize=(40,40))
tree.plot_tree(newmodel,feature_names=fn,class_names=cl,filled=True,fontsize=12);
fig.savefig('tree.png')

score = newmodel.score(test_X,test_Y)
print(score*100)

#confusion matrix
ypredit = newmodel.predict(test_X)
confusion_matrix(test_Y,ypredit)

plot_confusion_matrix(newmodel,test_X,test_Y,display_labels=['benign','malignant'])
