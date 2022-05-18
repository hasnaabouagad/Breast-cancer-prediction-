# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,plot_confusion_matrix,classification_report,log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_excel('breast-w-excelNombre.xlsx')

g=df.drop('Class', axis=1, inplace=False)

X_train,X_test,Y_train,Y_test= train_test_split(df[g.columns],df['Class'],random_state=0, test_size=0.1)

model=LogisticRegression()

model.fit(X_train,Y_train)

ypred = model.predict(X_test)
print(ypred)

confusion_matrix(Y_test,ypred)

plot_confusion_matrix(model,X_test,Y_test,display_labels=['benign','malignant'])

score = model.score(X_test,Y_test)
print(score*100)

probabilty=model.predict_proba(X_test)
print(ypred)
for prob in probabilty:
    print (list(map('{:.2f}'.format,prob)))

p1=model.predict_proba(X_train)
log_loss(Y_train,p1)

p2=model.predict_proba(X_test)
log_loss(Y_test,p2)
