import sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

df=pd.read_csv('newTrain1.csv')
drop=['Head','Id','Hazard']
y=df['Hazard']
for x in drop:
	df=df.drop(x,1)
x=np.array(df)
t=x
clf=ExtraTreesClassifier(n_jobs=4,class_weight='auto')
clf.fit(x,y)
df=pd.read_csv('newTest1.csv')
pid=df['Id']
drop=['Head','Id']
for x in drop:
	df=df.drop(x,1)
x=np.array(df)
test=x
output=clf.predict(test)
print clf.feature_importances_
f=open('output1.csv','w')
for i in range(len(output)):
	line= str(pid[i])+','+str(output[i])+'\n'
	f.write(line)