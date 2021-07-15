# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 07:51:16 2021

@author: ADITYA
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

data = pd.read_csv("creditcard.csv")
data.head()




#Make a bar plot of Class
pd.value_counts(data['Class']).plot.bar()
plt.title("Histogram of Class")
plt.xlabel("Class")
plt.ylabel("Count")

data['Class'].value_counts()


#Standardize the amount column 
from sklearn.preprocessing import StandardScaler
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Time','Amount'], axis = 1)
data.head()

X = np.array(data.loc[:,data.columns != 'Class'])
y = np.array(data.loc[:,data.columns == 'Class'])
print('Shape of X: {}'.format(X.shape))
print("Shape of y: {}".format(y.shape))


# The data for class is clearly imbalanced. We need to balance the Class column.
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

print("Shape of X_train dataset: ", (X_train.shape))
print("Shape of X_test dataset: ", (X_test.shape))
print("Shape of y_test dataset: ", (y_test.shape))
print("Shape of y_train dataset: ", (y_train.shape))

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report

parameters = {
    'C': np.linspace(1, 10, 10)
             }
lr = LogisticRegression()
clf = GridSearchCV(lr,parameters,cv = 5, verbose = 5, n_jobs = 3, return_train_score = True)
clf.fit(X_train_res,y_train_res.ravel())
clf.best_params_

#Build the Logistic Regression function
lr1= LogisticRegression(C = 9, verbose = 5.)
lr1.fit(X_train_res, y_train_res.ravel())

# make prediction on the y_train set
y_train_pre = lr1.predict(X_train)
cnf_matrix_tra = confusion_matrix(y_train,y_train_pre)

accuracy_y_train = (193971+318)/(193971+318+27+5048)
print(accuracy_y_train)

y_pre = lr1.predict(X_test)
cnf_matrix = confusion_matrix(y_test,y_pre)

accuracy_y_test = (83193+135)/(2103+12+135+83193)
print(accuracy_y_test)

tmp = lr1.fit(X_train_res,y_train_res.ravel())
y_pred_sample_score = tmp.decision_function(X_test)
fpr,tpr,tresholds = roc_curve(y_test,y_pred_sample_score)
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()