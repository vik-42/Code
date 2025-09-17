#The following code uses the Random Forest classifier implementation in scikitlearn to sort the most important attributes in the wine database

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium','Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', '0D280/0D315 of diluted wines','Proline']

X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)


forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs= -1) #n_jobs = -1 will use all processors, it represents the number of cuncurrent jobs
#note it is better to not use standardized datasets when working with randomized forests 
forest.fit(X_train,y_train)
feat_labels = df_wine.columns[1:]

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (i+1, 30, feat_labels[indices[i]], importances[indices[i]]))

plt.bar(range(X_train.shape[1]), importances[indices], color='blue',align='center')
plt.xticks(range(X_train.shape[1]),feat_labels[indices],rotation=90)
plt.tight_layout()
plt.show()