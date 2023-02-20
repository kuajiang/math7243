import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# init data
X1 = np.array([[3.81, -0.55],[0.23, 3.37],[3.05, 3.53],[0.68, 1.84],[2.67, 2.74]])
X2 = np.array([[-2.04,-1.25],[-0.72, -3.35],[-2.46,-1.31],[-3.51, 0.13],[-2.05, -2.82]])
X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((np.ones(X1.shape[0]), np.zeros(X2.shape[0])))

mu1 = np.mean(X1, axis=0)
mu2 = np.mean(X2, axis=0)
cov1 = np.cov(X1.T)
cov2 = np.cov(X2.T)
p1 = 0.5
p2 = 0.5

def qda_discriminant(x, mu, cov,p):
    d = x.shape[0]
    cov_inv = np.linalg.inv(cov)
    return -0.5*np.log(np.linalg.det(cov)) - 0.5*(x - mu).T.dot(cov_inv).dot(x - mu)+np.log(p)


# meshgrid
x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# predict
Z = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x = np.array([xx[i,j], yy[i,j]])
        d1 = qda_discriminant(x, mu1, cov1, p1)
        d2 = qda_discriminant(x, mu2, cov2, p2)
        if d1 > d2:
            Z[i,j] = 1
        else:
            Z[i,j] = 0

# plot
cmap = ListedColormap(['#FFAAAA', '#AAAAFF'])
plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)
plt.scatter(X1[:, 0], X1[:, 1], c='r', label='Label 1')
plt.scatter(X2[:, 0], X2[:, 1], c='b', label='Label 2')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Quadratic Discriminant Analysis')
plt.legend()
plt.show()