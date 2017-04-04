import sklearn
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
#from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
axis=35

df=pd.read_csv('train.csv')
y=df.ix[:,0]
x=df.ix[:,1:785]
y=np.array(y)
x=np.array(x)
pca=PCA(n_components=axis,whiten='True')
#x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=.30,random_state=33)
pca.fit(x)
x_train=pca.transform(x)
#clf=RandomForestClassifier(n_estimators=35,criterion='entropy')
#value=[.001,.003,.01,.03,.1,.3,1,3,10,30]
#for c in value:
c=30
clf=SVC(C=.01,gamma=.03)
clf.fit(x_train,y)
df=pd.read_csv('test.csv')
df=np.array(df)
df=pca.transform(df)
output=clf.predict(df)
print len(output)
#print clf.support_vectors_,clf.n_support_
fp=open('forest.txt','w')
fp.write('ImageId'+','+'Label'+'\n')
for i in range(0,28000):
	fp.write(str(i+1)+','+str(output[i])+'\n')

