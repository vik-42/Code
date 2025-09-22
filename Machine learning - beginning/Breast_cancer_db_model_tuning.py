import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

X = df.loc[:,2:].values
y = df.loc[:,1].values
le = LabelEncoder() #Econding the labelsn is necessary as the labels are not integers
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 1)

pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)),('clf', LogisticRegression(random_state=1))])

#The following is a less efficent approach, in here only to explain the algorithm
kfold = StratifiedKFold(shuffle=True, n_splits= 10, random_state = 1)
scores = []
for k, (train, test) in enumerate(kfold.split(X_train,y_train)):
    pipe_lr.fit(X_train[train],y_train[train]) # Train the model using only the rows (samples) indexed by 'train' in this fold
    score = pipe_lr.score(X_train[test], y_train[test]) # Evaluate the model on the test samples of this fold (unseen during training)
    scores.append(score)

#The following is a more efficient approach
scores_more_efficient = cross_val_score(estimator = pipe_lr, X = X_train, y=y_train, cv = 10, n_jobs = -1)
print('CV accuracy scores: %s' % scores_more_efficient)

# Learning curves show training vs validation performance as sample size grows.
# - Underfitting (high bias): both training and validation scores remain low.
# - Overfitting (high variance): training score is high, but validation score is much lower.

pipe_lr = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(penalty='l2', random_state=0))])
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1,1,10), cv=10, n_jobs = -1) #cv is the number of splits to be performed

train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

plt.plot(train_sizes,train_mean, color='blue', marker = 'o',linestyle ='--', label = 'training accuracy')
plt.plot(train_sizes, test_mean, color ='green', marker = 'x', linestyle='--', label = 'training variance')
plt.fill_between(train_sizes, train_mean+train_std,train_mean-train_std, label = 'validation accuracy', alpha = 0.15)
plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, color='green', alpha = 0.15, label = 'validation variance')
plt.grid()
plt.xlabel('# of training samples')
plt.ylabel('Accuracy')
plt.ylim([0.8,1.0])
plt.legend(loc ='lower right')
plt.show()

#The graph shows a slight difference between the training and the validation training curve, this indicates slight overfitting on training data, as they don't converge completely