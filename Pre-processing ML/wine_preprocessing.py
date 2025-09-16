from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium','Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', '0D280/0D315 of diluted wines','Proline']

X, y = df_wine.iloc[:,1:].values , df_wine.iloc[:,0].values

#test size will be 30% of the total, usually 30-70, 40-60 or 20-80 partitions are used between the training set and the test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=0)

#Standardization (scale 0-1) of the training and test data is necessary to avoid prioritizing certain attributes
stdsclr = StandardScaler()
X_train_std = stdsclr.fit_transform(X_train)
X_test_std = stdsclr.fit_transform(X_test)

lr = LogisticRegression(penalty='l1', C=0.1, solver='liblinear') #to use L2 penalty (less sparse solutions) delete the solver, it will output a 1 test accuracy -> indication of overfitting
#C value is the inverse of the regularization strenght, low C values mean the penalty term in the cost function is larger relative to the data fit term (higher bias, lower viariance, risk underfitting)
lr.fit(X_train_std,y_train)

print('train accuracy: %f'%lr.score(X_train_std,y_train))
print('test accuracy: %f'%lr.score(X_test_std,y_test))



