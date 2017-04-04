# coding: utf-8
execfile('script.py')
import xgboost xgb
import xgboost as xgb
df
label
test
y_true
clf=xgb.XGBClassifier(n_estimators=500,learning_rate=.02)
clf.fit(df,label
)
output=clf.predict(test)
accuracy_score(y_true,output)
est=[100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
rate=[.01,.02,.03,.1,.2,.3]
sub=[.80,.85,.90,.95.,1]
sub=[.80,.85,.90,.95,.1]
for a in est:
    for b in rate:
        for c in sub:
            clf=xgb.XGBClassifier(n_estimators=a,learning_rate=b,subsample=c,colsample_bytree=c)
            clf.fit(df,label)
            output=clf.predict(test)
            print accuracy_score(y_true,output)
            
for a in est:
    for b in rate:
        for c in sub:
            clf=xgb.XGBClassifier(n_estimators=a,learning_rate=b,subsample=c,colsample_bytree=c)
            clf.fit(df,label)
            output=clf.predict(test)
            print accuracy_score(y_true,output)
            
sub
sub=[.80,.85,.90,.95,1]
for a in est:
    for b in rate:
        for c in sub:
            clf=xgb.XGBClassifier(n_estimators=a,learning_rate=b,subsample=c,colsample_bytree=c)
            clf.fit(df,label)
            output=clf.predict(test)
            print accuracy_score(y_true,output)
            
results=[]
for a in est:
    for b in rate:
        for c in sub:
            clf=xgb.XGBClassifier(n_estimators=a,learning_rate=b,subsample=c,colsample_bytree=c)
            clf.fit(df,label)
            output=clf.predict(test)
            print accuracy_score(y_true,output)
            results.append((accuracy_score(y_true,output),a,b,c))
            
max(results)
max.sort()
max.sort
results.sort()
results
results.sort(desc=True)
results.sort('desc'=True)
results.sort(asce=False)
results.sort(reverse=True)
results
results[1:10]
results[1:20]
results[1:25]
results[1:40]
df
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
pca=PCA(n_components=5,whiten=False)
pca.fit(df)
pca.explained_variance_ratio_
new_train=pca.transform(df)
new_train.shape
new_train
for x in range(len(new_train)):
    if(label(x)==0):
        color='red'
    else:
        color='blue'
    plt.scatter(new_train[x][0],new_train[x][1],c=color)
    
for x in range(len(new_train)):
    if(label[x]==0):
        color='red'
    else:
        color='blue'
    plt.scatter(new_train[x][0],new_train[x][1],c=color)
    
plt.show()
for x in range(len(new_train)):
    if(label[x]==0):
        color='red'
    else:
        color='blue'
    plt.scatter(new_train[x][0],new_train[x][1],new_train[x][2],c=color)
    
plt.show()
for x in range(len(new_train)):
    if(label[x]==0):
        color='red'
    else:
        color='blue'
    mplt3d.scatter(new_train[x][0],new_train[x][1],new_train[x][2],c=color)
    
for x in range(len(new_train)):
    if(label[x]==0):
        color='red'
    else:
        color='blue'
    mplot3d.scatter(new_train[x][0],new_train[x][1],new_train[x][2],c=color)
    
from matplotlib import *
for x in range(len(new_train)):
    if(label[x]==0):
        color='red'
    else:
        color='blue'
    mplot3d.scatter(new_train[x][0],new_train[x][1],new_train[x][2],c=color)
    
from matplotlib import mplofrom mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = Axes3D(fig)
for x in range(len(new_train)):
    if(label[x]==0):
        color='red'
    else:
        color='blue'
    ax.scatter(new_train[x][0],new_train[x][1],new_train[x][2],c=color)
    
plt.show()
clf
xgb.plotting(clf)
xgb.plot_tree(clf)
xgb.plot_importance
xgb.plot_importance()
xgb.plot_importance(clf)
plt.show()
plt.show()
xgb.plot_importance(clf)
plt.show()
clf
execfile('script.py')
clf.scor
clf.score
clf.score()
clf.score(test)
clf.score(test,y_true)
clf.score(test,y_true)
clf.score(test,y_true)
clf.best_estimator_
model=clf.best_estimator_
model.fit(df,label)
output=model.predict(test)
accuracy_score(y_true,output)
accuracy_score(y_true,output)
clf.grid_scores_
clf.refit
clf.refit()
clf.refit
clf.fit(X,y)
output=clf.predict(test)
accuracy_score(y_true,output)
clf
clf.best_score_
clf.best_params_
clf.best_estimator_
output=clf.best_estimator_.predict(test)
accd
accuracy_score(y_true,output)
model=clf.best_estimator_
model.feature_importances_
model.tree_
model.tree_()
model.tree_
x_min,x_max=X[:,0].min()-1,X[:,0].max()+!
x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
xx,yy=np.meshgrid(np.arrang(x_min,x_max,.02),np.arrange(y_min,y_max,.02))
xx,yy=np.meshgrid(np.arrange(x_min,x_max,.02),np.arrange(y_min,y_max,.02))
xx,yy=np.meshgrid((x_min,x_max,.02),(y_min,y_max,.02))
xx.ravel()
xx
xx.ravel()
plt.contourf(xx,yy,cmap=plt.cm.Paired)
plt.contourf(xx,yy,1,cmap=plt.cm.Paired)
plt.contourf(xx,yy,[[1]],cmap=plt.cm.Paired)
plt.contourf(xx,yy,,cmap=plt.cm.Paired)
model
Z=moodel.predict(np.c_[xx.ravel(),yy.ravel()])
Z=model.predict(np.c_[xx.ravel(),yy.ravel()])
Z=model.predict(np.c_[xx.ravel(),yy.ravel()])
Z=model.predict(np.c_[xx.ravel(),yy.ravel()])
model
Z=model.predict(np.c_[xx.ravel(),yy.ravel()])
xx.ravel
xx.ravel(),yy.ravel()
[xx.ravel(),yy.ravel()]
np.c_[xx.ravel(),yy.ravel()]
Z=model.predict([xx.ravel(),yy.ravel()])
plt.contourf(xx,yy,Z,cmap=plt.cm.Paired)
plt.contourf(xx,yy,Z,cmap=plt.cm.Paired)
len(x)
len(Z)
len(Z[0])
Z
plt.contourf(xx,yy,Z,cmap=plt.cm.Paired)
Z=Z.reshape(xx.shape)
xx.shape
Z
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
xx
Z=model.predict(np.c_[xx.ravel(),yy.ravel()])
Z=model.predict([xx.ravel(),yy.ravel()])
Z=model.predict(np.c_[xx.ravel(),yy.ravel()])
Z=model.predict(np.c_[xx.ravel(),yy.ravel()])
xx.ravel()
xx.ravel(),yy.ravel()
np.c_(xx.ravel(),yy.ravel())
np.c_[xx.ravel(),yy.ravel()]
[xx.ravel(),yy.ravel()]
Z=model.predict([xx.ravel(),yy.ravel()])
xx.ravel()
len(xx.ravel())
Z=model.predict([xx.ravel(),yy.ravel()])
clf=DecisionTreeClassifier()
clf.fit(df,y)
output=clf.predict(test)
accuracy_score(y_true,output)
clf
X
clf=DecisionTreeClassifier()
clf.fit(X,y)
output=clf.predict(test)
accuracy_score(y_true,output)
x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
x_min
x_max
y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
Z=clf.predict(np.c_[xx,yy])
Z=clf.predict(XX)
Z=clf.predict(xx)
x
xx
xx
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
combined
for x in combined:
    print x
    
np.sum(combined,axis=1)
combined.shape
len(combined)
np.sum(combined,axis=0)
test.shape
len(np.sum(combined,axis=0))
[lambda x: x*x for x in range(10)]
output=np.sum(combined,axis=0)
output
lambda x: for x in output
[lambda x: for x in output]
[lambda x: x for x in output]
[lambda x: y for y in output]
(lambda x: x)(x) for x in output
[(lambda x: x)(x) for x in output]
input
input=[]
[(lambda x: input.append(1) if x > 10 else input.append(0))(x) for x in output]
input
output
input
accuracy_score(y_true,input)
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('decisionTree.py 20')
execfile('decisionTree.py',20)
execfile('decisionTree.py ,20')
execfile('decisionTree.py 20')
execfile('decisionTree.py')
import sys
sys.argv=10
execfile('decisionTree.py')
sys.argv=['10']
execfile('decisionTree.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
max_val
max_val.sort(reverse=True)
max_val
max_val[1]
max_val[0][1]
max_val[0][0]
temp=max_val[0][1]
temp
import sklearn.tree
sklearn.tree.export_graphviz(final_model)
sklearn.tree.export_graphviz(temp)
sklearn.tree.export_graphviz(temp)
sklearn.tree.export_graphviz(temp,file='temp')
sklearn.tree.export_graphviz(temp,out_file='tree.dot')
sklearn.tree.export_graphviz(temp,out_file='tree.dot')
temp
temp.fit(X,y)
sklearn.tree.export_graphviz(temp,out_file='tree.dot')
temp.fit(X,y)
temp.fit(X,y)
temp.fit(X,y)
temp.fit(X,y)
temp.fit(X,y)
temp.fit(X,y)
sklearn.tree.export_graphviz(temp,out_file='tree.dot')
temp.predict(test)
sklearn.tree.export_graphviz(temp,out_file='tree.dot')
temp.predict(test)
from sklearn import tree
temp.fit(X,y)
tree.export_graphviz(temp,out_file='tree.dot')
tree.export_graphviz(temp,out_file='tree.dot','r')
tree.export_graphviz(temp,out_file='tree.dot')
clf
clf=DecisionTreeClassifier()
clf.fit(X,y)
tree.export_graphviz(clf,out_file='tree.dot')
tree.export_graphviz(clf,out_file='tr.dot')
tree.export_graphviz(clf,out_file='tr.dot')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
final_model
final_model.tree
final_model.tree_
s=final_model.tree_
s
tree.export_graphviz(s,out_file='tr.dot')
tree.export_graphviz(s,out_file='tr.dot')
s
final_model
final_model.feature_importances_
tree.export_graphviz(final_model)
import os
os.system('ls -l')
name=tree.dot
name='tree.dot'
name.replace('.dot','.png')
x for x in range(100,1000,50)
x for x in range(100,1000,50):
    
[x for x in range(100,1000,50)]
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
max_val
max_val[0]
execfile('script.py')
max_val[0]
max_val[1]
max_val[2]
max_val
max_val[0]
max_val[0:10]
max_val[0:25]
execfile('script.py')
max_val[0:25]
execfile('script.py')
max_val[0:25]
execfile('script.py')
max_val[0:25]
execfile('script.py')
max_val[0:4025]
max_val[0:40]
execfile('script.py')
max_val[0:25]
execfile('script.py')
max_val[0:25]
30*.90625
execfile('script.py')
30*.90625
max_val[0:25]
max_val[0]
max_val[0][1]
s=max_val[0][1].predict(X)
accuracy_score(label,s)
s=max_val[0][1].predict(test)
accuracy_score(label,s)
accuracy_score(y_true,s)
df=pd.read_csv('combined.csv')
df
df
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
LabelEncoder.fit(df.Class)
df=pd.read_csv('combined.csv')
df
df.shape[0]
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
("")
execfile('script.py')
execfile('script.py')
execfile('script.py')
all_entries.shape
tr
ts
tr
all_entries[tr]
all_entries[tr,:]
type(all_entries)
execfile('script.py')
execfile('script.py')
clf
all_entries[ts]
clf.predict(all_entries[ts])
output=clf.predict(all_entries[ts])
output
execfile('script.py')
execfile('script.py')
execfile('script.py')
kf
for tr,ts in kf:
    print tr,ts
    
execfile('script.py')
for tr,ts in kf:
    print tr,ts
    
execfile('script.py')
execfile('script.py')
max_val.sort(reverse=True)
max_val[1:10]
execfile('script.py')
execfile('script.py')
execfile('script.py')
max_val[0]
execfile('script.py')
max_val[0]
max_val[0].predict(test)
max_val[0][1].predict(test)
output=max_val[0][1].predict(test)
accuracy_score(y_true,output)
execfile('script.py')
execfile('script.py')
execfile('script.py')
execfile('script.py')
mean_score
len(max_val)
len(mean_score)
execfile('script.py')
max_val[2*number_of_folds][0]
execfile('script.py')
execfile('script.py')
1.7272
1.7272/2
.8181+
.8181+.9090
(.8181+.9090)/2
max_index
len(max_index)
len(mean_score)
execfile('script.py')
max_index
max_index*number_of_folds
best_model
execfile('script.py')
execfile('script.py')
get_ipython().magic(u'save 1-428')
