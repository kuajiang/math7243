import io

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import t

url = "https://raw.githubusercontent.com/tipthederiver/Math-7243-2020/master/Datasets/Ames/train.csv"
s = requests.get(url).content
ames = pd.read_csv(io.StringIO(s.decode('utf-8')))

columns = list(ames.columns)

z = ames['GrLivArea']+ames['BsmtUnfSF']<4000

print("Number of records removed:",len(ames) - sum(z))
data = ames[z]
data = data.select_dtypes(include=['int64','float64'])
data = data.drop(columns='MSSubClass')
print("data:", data.shape)

test_size = int(data.shape[0]*0.8)
print("test_size",test_size)

N = 1000

beta0 = np.zeros(N)
beta1 = np.zeros(N)

for i in range(N):
    ## Compute beta0 and beta1, using linear algebra, sklearn, or scipy
    train=data.sample(n=test_size,replace=True)
    test=data.drop(train.index)
    
    X_train = train.drop(columns=['SalePrice','Id'])
    Y_train = train['SalePrice']

    X = np.matrix(X_train['1stFlrSF'])
    Y = np.matrix(Y_train)

    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    
    Xa = np.append(np.ones(X.shape),X,1)
    betas = (Xa.T*Xa).I*Xa.T*Y
    
    beta0[i] = betas[0]
    beta1[i] = betas[1]

# plot the first histogram
plt.subplot(1, 2, 1)
plt.hist(beta0, bins=15, color='blue', alpha=0.7)
plt.xlabel('beta0')
plt.ylabel('Frequency')
plt.title('Histogram of beta0')

# plot the second histogram
plt.subplot(1, 2, 2)
plt.hist(beta1, bins=15, color='red', alpha=0.7)
plt.xlabel('beta1')
plt.ylabel('Frequency')
plt.title('Histogram of beta1')

# show the plots
plt.tight_layout()
plt.show()

beta0.sort()

interval = (beta0[25],beta0[975])
print("beta0 95%% interval",interval)


X = np.matrix(data['1stFlrSF']).reshape(-1,1)
Y = np.matrix(data['SalePrice']).reshape(-1,1)
Xa = np.append(np.ones(X.shape),X,1)


beta = (Xa.T*Xa).I*Xa.T*Y

def RSS(y,Y):
    y = np.matrix(y).reshape(-1,1)
    Y = np.matrix(Y).reshape(-1,1)
    return (y-Y).T*(y-Y)


rss = RSS(Xa*beta, Y).A1
df = X.shape[0] - X.shape[1] -1
sd = np.sqrt(rss/df)
print("sd",sd)


quantile = t.ppf(0.975, df)
print("quantile", quantile)


half_range = quantile*sd*np.sqrt(np.diagonal((Xa.T*Xa).I)).reshape((-1,1))
print("half_range", half_range)

lower = beta - half_range
upper = beta + half_range
print("lower", lower)
print("upper", upper)