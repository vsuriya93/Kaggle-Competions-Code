# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn import tree
import sys
# training set
df=pd.read_csv('train.csv')
label=df.Class
# test set
test=pd.read_csv('test.csv')
y_true=test.Class
# remove unwanted columns
drop_columns=['Bank','Class']
for col_name in drop_columns:
            df.drop(col_name,axis=1,inplace=True)
            test.drop(col_name,axis=1,inplace=True)
    
# convert dataframe to array
X=np.array(df)
y=np.array(label)
crit=['gini']
max_depth=[5,6,7,8,9]
min_sample=[1,2]
max_val=[]
for a in crit:
            for b in max_depth:
                        for c in min_sample:
                                    clf=DecisionTreeClassifier(criterion=a,max_depth=b,min_samples_split=c)
                                    clf.fit(X,y)
                                    output=clf.predict(test)
                                    max_val.append((accuracy_score(y_true,output),clf))
                                    #print accuracy_score(y_true,output),a,b,c
            
max_val.sort(reverse=True)
final_model=max_val[0][1]
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
plot_step=.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
xx
yy
clf
clf.predict(np.c_[xx.ravel() yy.ravel()])
clf.predict(np.c_[xx.ravel(), yy.ravel()])
clf.predict(np.c_[xx.ravel(), yy.ravel()])
np.c_
np.c_([xx.ravel() yy.ravel()],2)
np.c_([xx.ravel(), yy.ravel()],2)
np.c_[[xx.ravel(), yy.ravel()],2]
get_ipython().magic(u'pinfo2 c_')
get_ipython().magic(u'pinfo2 np.c_')
np.r_[[xx.ravel(), yy.ravel()],2]
np.r_[xx.ravel(), yy.ravel()]
len(np.r_[xx.ravel(), yy.ravel()])
len(np.c_[xx.ravel(), yy.ravel()])
get_ipython().magic(u'pinfo2 np.ravel')
xx
len(xx)
len(xx[0])
X
X
df=pd.read_csv('train.csv')
label=df.Class
# test set
test=pd.read_csv('test.csv')
y_true=test.Class
# remove unwanted columns
drop_columns=['Bank','Class']
for col_name in drop_columns:
            df.drop(col_name,axis=1,inplace=True)
            test.drop(col_name,axis=1,inplace=True)
    
# convert dataframe to array
X=np.array(df)
y=np.array(label)
crit=['gini']
max_depth=[5,6,7,8,9]
min_sample=[1,2]
max_val=[]
for a in crit:
            for b in max_depth:
                        for c in min_sample:
                                    clf=DecisionTreeClassifier(criterion=a,max_depth=b,min_samples_split=c)
                                    clf.fit(X,y)
                                    output=clf.predict(test)
                                    max_val.append((accuracy_score(y_true,output),clf))
                                    #print accuracy_score(y_true,output),a,b,c
            
max_val.sort(reverse=True)
final_model=max_val[0][1]
x
X.shape

############################################################ Actual Script #######################################
from sklearn.decomposition import PCA
pca=PCA(n_components=2,whiten=False)
pca.fit_transform(X)
X
X=pca.fit_transform(X)
X
clf=DecisionTreeClassifier()
clf.fit(X,y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
xx
yy
len(yy)
len(yy[0])
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z
Z=Z.reshape(xx.shape)
Z
import matplotlib.pyplot as plt
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
cs
plt.show()
for i in range(len(X)):
    if label[i]==0:
        color='red'
    else :
        color='blue'
    plt.scatter(X[i][0],X[i][1],c=color)
    
plt.show()
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
for i in range(len(X)):
    if label[i]==0:
        color='red'
    else :
        color='blue'
    plt.scatter(X[i][0],X[i][1],c=color)
    
plt.show()
test=pca.transform(test)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
for i in range(len(X)):
    if label[i]==0:
        color='red'
    else :
        color='blue'
    plt.scatter(test[i][0],test[i][1],c=color)
    
for i in range(len(test)):
    if label[i]==0:
        color='red'
    else :
        color='blue'
    plt.scatter(test[i][0],test[i][1],c=color)
    
plt.show()
get_ipython().magic(u'save tree_scripy.py 1-107')
