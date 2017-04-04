import sklearn
import pandas as pd
import numpy as np
from sklearn import svm

df=pd.read_csv('newTrain.csv')
drop=['Head','Id','Hazard']
y=df['Hazard']
for x in drop:
	df=df.drop(x,1)
x=np.array(df)
t=x
clf=svm.SVR()
clf.fit(x,y)
df=pd.read_csv('newTest.csv')
pid=df['Id']
drop=['Head','Id']
for x in drop:
	df=df.drop(x,1)
x=np.array(df)
test=x
output=clf.predict(test)
f=open('output.csv','w')
for i in range(len(output)):
	line= str(pid[i])+','+str(output[i])+'\n'
	f.write(line)