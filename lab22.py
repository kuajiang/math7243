import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import statsmodels.api as sm

file_dir = './MRI_Images/'

labels = pd.read_csv(file_dir + 'labels.csv')
# display(labels)

DS = 8             # Downsample rate, must be a multiple of 30976

if 30976/DS % 1 > 0:
    print("Downsample rate is not a multiple of 30976")
    DS = 1
    im_size = 30976
else:
    im_size = int(30976/DS)


data = np.zeros([609, im_size])

for i, file_name in enumerate(labels.Filename):
    img = np.mean(matplotlib.image.imread(file_dir + file_name),axis=2).reshape(-1)
    data[i,:] = img[::DS]            # Downsample the image

test_size = int(data.shape[0]*0.8)
data = pd.DataFrame(data)
train = data.sample(n=test_size,replace=False,random_state=255)
test = data.drop(train.index)

y = labels["nWBV"]

def RSS(y,Y):
    y = np.matrix(y).reshape(-1,1)
    Y = np.matrix(Y).reshape(-1,1)
    
    return (y-Y).T*(y-Y)

def RMS(y,Y):    
    return np.sqrt(RSS(y,Y))/len(y)

def Rs(y,Y): 
    y = np.matrix(y).reshape(-1,1)
    Y = np.matrix(Y).reshape(-1,1)
        
    return 1 - RSS(y,Y)/((Y - np.mean(Y)).T*(Y - Y.mean()))

r_sqds = []
alphas = np.linspace(0, 0.02,21)

for alp in alphas:
    ols = sm.OLS(Y_train, sm.add_constant(X_train))
    ols_result = ols.fit_regularized(L1_wt=0,alpha=alp)
    
    y_pred = ols_result.predict(sm.add_constant(X_test, has_constant="add"))
    score = Rs(y_pred,Y_test).A1
    print(alp, score)
    r_sqds.append(score)

max_index = np.argmax(r_sqds)
alpha_max_sqds = alphas[max_index]
print(np.max(r_sqds), alpha_max_sqds)

plt.plot(alphas, r_sqds)





###### Lasso ##########


DS = 32             # Downsample rate, must be a multiple of 30976

if 30976/DS % 1 > 0:
    print("Downsample rate is not a multiple of 30976")
    DS = 1
    im_size = 30976
else:
    im_size = int(30976/DS)


data = np.zeros([609, im_size])

for i, file_name in enumerate(labels.Filename):
    img = np.mean(matplotlib.image.imread(file_dir + file_name),axis=2).reshape(-1)
    data[i,:] = img[::DS]            # Downsample the image

test_size = int(data.shape[0]*0.8)
data = pd.DataFrame(data)
train = data.sample(n=test_size,replace=False,random_state=255)
test = data.drop(train.index)

y = labels["nWBV"]

X_train = train
Y_train = y[train.index]

X_test = test
Y_test = y[test.index]

r_sqds = []
alphas = np.linspace(0, 0.01,11)

for alp in alphas:
    ols = sm.OLS(Y_train, sm.add_constant(X_train))
    ols_result = ols.fit_regularized(L1_wt=1,alpha=alp)    
    
    y_pred = ols_result.predict(sm.add_constant(X_test, has_constant="add"))
    score = Rs(y_pred,Y_test).A1
    print(alp, score)
    r_sqds.append(score)

max_index = np.argmax(r_sqds)
alpha_max_sqds = alphas[max_index]
print(np.max(r_sqds), alpha_max_sqds)

plt.plot(alphas, r_sqds)

