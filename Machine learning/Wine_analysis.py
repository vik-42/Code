from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium','Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', '0D280/0D315 of diluted wines','Proline']

X, y = df_wine.iloc[:,1:].values , df_wine.iloc[:,0].values

#test size will be 30% of the total, usually 30-70, 40-60 or 20-80 partitions are used between the training set and the test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=0)

#Standardization (scale 0-1) of the training and test data is necessary to avoid prioritizing certain attributes
stdsclr = StandardScaler()
X_train_std = stdsclr.fit_transform(X_train)
X_test_std = stdsclr.fit_transform(X_test)

lr = LogisticRegression(penalty='l1', C=0.1, solver='saga') #to use L2 penalty (less sparse solutions) delete the solver, it will output a 1 test accuracy -> indication of overfitting
#C value is the inverse of the regularization strenght, low C values mean the penalty term in the cost function is larger relative to the data fit term (higher bias, lower viariance, risk underfitting)
lr.fit(X_train_std,y_train)

print('train accuracy without sbs: %f'%lr.score(X_train_std,y_train))
print('test accuracy without sbs: %f'%lr.score(X_test_std,y_test))


#To better improve the model's capability to predict, a SBS algorithm could be implemented, to automatically reduce the number of didmensions to a more suitable one, in order to reduce noise.

acc = []
k_val = []
for k in range(1, len(X[1,:])):
    k_val.append(k)
    sfs = SequentialFeatureSelector(lr, n_features_to_select=k, direction ='backward')
    sfs.fit(X_train_std,y_train) #It is better to use scaled data
    X_train_scaled_sfs_bwd = sfs.transform(X_train_std)
    X_test_scaled_sfs_bwd = sfs.transform(X_test_std)
    lr.fit(X_train_scaled_sfs_bwd, y_train)
    acc.append(lr.score(X_test_scaled_sfs_bwd,y_test))
    
#this code allows to evaluate the linear model's accuracy using more attributes
plt.plot(k_val,acc, marker = 'o')
plt.grid()
plt.show()




