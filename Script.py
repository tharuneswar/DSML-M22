import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------
# Reading CSV

os.chdir('C:\\Users\\Karan\\Desktop\\Tharun\\DSML\\Res\\Datasets_final')

data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')

df= data.copy()

#Some Basic EDA

col_lst = df.columns

df_des = df.describe()

print(df.info())

Corr_mat = df.corr()

#-------------------------------------------------------------------------------



X = df[['CD Account','Income']]

y = df['CreditCard']

X.info()


# Splitting the dataset into the Training set and Test set    -----------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#-------------------------------------LR---------------------------------------

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(random_state = 0,C =1000)
classifier_LR.fit(X_train, y_train)

# Predicting the Test set results
y_pred_LR = classifier_LR.predict(X_test)

print('-------------------------')
print('-------------------------')
print('LOGISTIC REGRESSION')
print('-------------------------')
print('-------------------------')

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(y_test, y_pred_LR)
print('--------------------')
print('| Confusion Matrix |')
print('--------------------')
print(cm_LR)

# get classification report

from sklearn import metrics
classification_report_LR = metrics.classification_report(y_test, y_pred_LR)
print('-------------------------')
print('| Classifiction Report |')
print('-------------------------')
print(classification_report_LR)


from sklearn import metrics
Metrics_LR = metrics.accuracy_score(y_test, y_pred_LR)
print('-------------------------')
print('| Accuracy Score |')
print('-------------------------')
print(Metrics_LR)

from sklearn.model_selection import cross_val_score
CVS_LR = cross_val_score(classifier_LR,X,y, cv = 10).mean()
print('-------------------------')
print('| Cross Validation Score |')
print('-------------------------')
print(CVS_LR)

#-------------------------------KNN-------------------------------------------

# Fitting KNN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN =  KNeighborsClassifier(n_neighbors=3) 
classifier_KNN.fit(X_train, y_train)

# Predicting the Test set results
y_pred_KNN = classifier_LR.predict(X_test)



print('-------------------------')
print('-------------------------')
print('KNN CLASSIFIER')
print('-------------------------')
print('-------------------------')

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_KNN = confusion_matrix(y_test, y_pred_KNN)
print('--------------------')
print('| Confusion Matrix |')
print('--------------------')
print(cm_KNN)

# get classification report
from sklearn import metrics
classification_report_KNN = metrics.classification_report(y_test, y_pred_KNN)
print('-------------------------')
print('| Classifiction Report |')
print('-------------------------')
print(classification_report_KNN)


from sklearn import metrics
Metrics_KNN = metrics.accuracy_score(y_test, y_pred_KNN)
print('-------------------------')
print('| Accuracy Score |')
print('-------------------------')
print(Metrics_KNN)

from sklearn.model_selection import cross_val_score
CVS_KNN = cross_val_score(classifier_KNN,X,y, cv = 10).mean()
print('-------------------------')
print('| Cross Validation Score |')
print('-------------------------')
print(CVS_KNN)

#-------------------------------RF---------------------------------------------

# Fitting RF to the Training set

from sklearn.ensemble import RandomForestClassifier 
classifier_RF =  RandomForestClassifier(max_depth=2, random_state=0) 
classifier_RF.fit(X_train, y_train)


# Predicting the Test set results
y_pred_RF = classifier_RF.predict(X_test)

print('-------------------------')
print('-------------------------')
print('RANDOM FORST CLASSIFIER')
print('-------------------------')
print('-------------------------')

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RF = confusion_matrix(y_test, y_pred_RF)
print('--------------------')
print('| Confusion Matrix |')
print('--------------------')
print(cm_RF)

# get classification report
from sklearn import metrics
classification_report_RF = metrics.classification_report(y_test, y_pred_RF)
print('-------------------------')
print('| Classifiction Report |')
print('-------------------------')
print(classification_report_RF)


from sklearn import metrics
Metrics_RF = metrics.accuracy_score(y_test, y_pred_RF)
print('-------------------------')
print('| Accuracy Score |')
print('-------------------------')
print(Metrics_RF)

from sklearn.model_selection import cross_val_score
CVS_RF = cross_val_score(classifier_RF,X,y, cv = 10).mean()
print('-------------------------')
print('| Cross Validation Score |')
print('-------------------------')
print(CVS_RF)