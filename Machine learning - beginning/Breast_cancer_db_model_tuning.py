import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

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

plt.plot(train_sizes,train_mean, color='blue', marker = 'o', label = 'training accuracy')
plt.plot(train_sizes, test_mean, color ='green', marker = 'x', linestyle='--', label = 'training variance')
plt.fill_between(train_sizes, train_mean+train_std,train_mean-train_std, label = 'training variance', alpha = 0.15)
plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, color='green', alpha = 0.15, label = 'validation variance')
plt.grid()
plt.xlabel('# of training samples')
plt.ylabel('Accuracy')
plt.ylim([0.8,1.0])
plt.legend(loc ='lower right')
plt.show()

#The graph shows a slight difference between the training and the validation training curve, this indicates slight overfitting on training data, as they don't converge completely

#We can now add validation curves to solve overfitting and underfitting problems by evaluating the C parameter in the logistic regression model:
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(estimator=pipe_lr, X=X_train, y = y_train, param_name='clf__C', param_range = param_range, cv = 10)
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

plt.plot(param_range,train_mean, color='blue', marker = 'o', label = 'training accuracy')
plt.plot(param_range, test_mean, color ='green', marker = 'x', linestyle='--', label = 'training variance')
plt.fill_between(param_range, train_mean+train_std,train_mean-train_std, label = 'training variance', alpha = 0.15)
plt.fill_between(param_range, test_mean+test_std, test_mean-test_std, color='green', alpha = 0.15, label = 'validation variance')
plt.xscale('log')
plt.grid()
plt.xlabel('# of training samples')
plt.ylabel('Accuracy')
plt.ylim([0.8,1.0])
plt.legend(loc ='lower right')
plt.show()

#There is a slight overfit when incrementing C over 0.1, which is the optimal value. To better optimize the C parameter Grid-Search is implemented (computationally expensive);
#A less expensive approach could be achieved by using RandomizedSearchCV;
#A Support Vector Machine model is used this time:

pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [ {'clf__C' : param_range , 'clf__kernel' : ['linear']},
              {'clf__C' : param_range , 'clf__gamma':param_range, 'clf__kernel':['rbf']}]
gs = GridSearchCV(estimator = pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10,n_jobs=-1)
gs = gs.fit(X_train, y_train)

print('The best score is %3f \n' %gs.best_score_)
print('The best parameters are %s'% gs.best_params_)


clf = gs. best_estimator_
clf.fit(X_train, y_train)
print('Test accurac of the best model: %.3f'%clf.score(X_train, y_train))

#As accuracy is not a reliable metrics, Precision, recall and F1-Score are implemented in the following:

#The confusion matrix is used to better understand the given results
#   0 1 (predicted)
#0
#1
#(real)
#1 = malign tumor
#0 = bening tumor

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test,y_pred=y_pred)
print(confmat)
