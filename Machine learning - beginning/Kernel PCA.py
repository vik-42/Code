#The following is just a simple Kernel PCA implementation using scikit-learn

from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X,y = make_moons(1000, random_state = 320)

scikit_kpca = KernelPCA(n_components=2,kernel='rbf', gamma = 15)
X_skernpca = scikit_kpca.fit_transform(X)

#plt.scatter(X_skernpca[:,0], X_skernpca[:,1], marker = 'x')
plt.scatter(X_skernpca[y==0,0], X_skernpca[y==0,1], alpha = 0.5, marker = 'o')
plt.scatter(X_skernpca[y==1,0], X_skernpca[y==1,1], alpha = 0.5, marker = '^')
plt.show()