import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

X1 = np.array([[3.81, -0.55],[0.23, 3.37],[3.05, 3.53],[0.68, 1.84],[2.67, 2.74]])
X2 = np.array([[-2.04,-1.25],[-0.72, -3.35],[-2.46,-1.31],[-3.51, 0.13],[-2.05, -2.82]])

def quadratify(X):
    x1 = X[:, 0].reshape((-1,1))
    x2 = X[:, 1].reshape((-1,1))
    return np.concatenate((X, x1*x1, x1*x2, x2*x2),axis=1)

X1 = quadratify(X1)
X2 = quadratify(X2)

X = np.concatenate((X1, X2), axis=0)

y = np.concatenate((np.ones(X1.shape[0]), np.zeros(X2.shape[0])))

mu1 = np.mean(X1, axis=0)
mu2 = np.mean(X2, axis=0)
p1 = 0.5
p2 = 0.5
cov = np.zeros((X.shape[1], X.shape[1]))

for x in X1:
    xdistance = (x-mu1).reshape((-1,1))
    cov += xdistance.dot(xdistance.T)
for x in X2:
    xdistance = (x-mu2).reshape((-1,1))
    cov += xdistance.dot(xdistance.T)
cov = cov/(len(X)-2)

inv_cov = np.linalg.inv(cov)

def lda_discriminant(x, mu, cov, p):
    return x.dot(inv_cov).dot(mu.T) - 0.5 * mu.dot(inv_cov).dot(mu.T)+np.log(p)


# meshgrid
x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# predict
Z = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x = np.array([xx[i,j], yy[i,j], xx[i,j]*xx[i,j], xx[i,j]*yy[i,j], yy[i,j]*yy[i,j]])
        d1 = lda_discriminant(x, mu1, cov, p1)
        d2 = lda_discriminant(x, mu2, cov, p2)
        if d1 > d2:
            Z[i,j] = 1
        else:
            Z[i,j] = 0

# plot
cmap = ListedColormap(['#FFAAAA', '#AAAAFF'])
plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)
plt.scatter(X1[:, 0], X1[:, 1], c='b', label='Label 1')
plt.scatter(X2[:, 0], X2[:, 1], c='r', label='Label 2')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('quadratic LDA')
plt.legend()
plt.show()
