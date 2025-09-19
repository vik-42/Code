import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium','Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', '0D280/0D315 of diluted wines','Proline']

sc = StandardScaler()

X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test) #be careful, avoid data leakage

cov_mat = np.cov(X_train_std.T)

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

#to now perform coviariance ratios

tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse = True)]
cumulative_var_exp = np.cumsum(var_exp)

plt.bar(range(1,14), var_exp, label='cumulative variance')
plt.step(range(1,14), cumulative_var_exp, where='mid')
plt.xlabel('principal components')
plt.show()

pca = PCA(n_components=4)
lr = LogisticRegression()

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr.fit(X_train_pca,y_train)

print('The logistic regression scored on training: %f'%lr.score(X_train_pca, y_train))
print('The logistic regression scored on test: %f'%lr.score(X_test_pca, y_test))







