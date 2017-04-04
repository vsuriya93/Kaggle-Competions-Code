import pandas as pd
import sklearn
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn import ensemble
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
df=pd.read_csv('train.csv')
dropValues=['Name','Cabin','PassengerId','Ticket','Sex','Embarked','Survived']
dummies=['Sex','Embarked']
for i in  dummies:
	temp=pd.get_dummies(df[i])
#	print temp
	for j in temp.columns.values:
		df[j]=temp[j]
label=df['Survived']
for i in dropValues:
	df=df.drop(i,1)
df['Age']=df['Age'].interpolate()
df['Fare']=df['Fare'].interpolate()
x=np.array(df)
y=np.array(label)
clf2=AdaBoostClassifier()
clf3=BaggingClassifier(AdaBoostClassifier())
clf4=RandomForestClassifier()
clf5=RandomForestRegressor()
clf6=GradientBoostingClassifier()
clf2.fit(x,y)
clf3.fit(x,y)
clf4.fit(x,y)
clf5.fit(x,y)
clf6.fit(x,y)
clf=ExtraTreesClassifier(n_estimators=35,criterion='entropy',n_jobs=4)
clf.fit(x,y)
df=pd.read_csv('test.csv')
dropValues=['Name','Cabin','PassengerId','Ticket','Sex','Embarked']
dummies=['Sex','Embarked']
for i in  dummies:
	temp=pd.get_dummies(df[i])
#	print temp
	for j in temp.columns.values:
		df[j]=temp[j]
pid=df['PassengerId']
for i in dropValues:
	df=df.drop(i,1)
df['Age']=df['Age'].interpolate()
df['Fare']=df['Fare'].interpolate()
x=np.array(df)
output1=clf3.predict(x)
output2=clf4.predict(x)
output3=clf5.predict(x)
output4=clf6.predict(x)
output5=clf.predict(x)
f=open('output.csv','w')
f.write('PassengerId,Survived\n')
c=892
output=[]
for i in range(len(output1)):
	if output1[i]+output2[i]+output3[i]+output4[i]+output5[i]>=3:
		output.append(1)
	else:
		output.append(0)
for i in range(len(output)):
	line= str(pid[i])+','+str(output[i])+'\n'
	f.write(line)