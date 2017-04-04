import pandas as pd
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

y=df['Survived']
pid=test['PassengerId']
df.drop('Survived',axis=1,inplace=True)

drop_columns=['Ticket','Cabin','SibSp','Parch','PassengerId','Embarked','Name']
cat_columns=['Sex','Embarked']

encoder=LabelEncoder()
encoder.fit(list(df['Sex'])+list(test['Sex']))
df['Sex']=encoder.transform(df['Sex'])
test['Sex']=encoder.transform(test['Sex'])

temp=pd.get_dummies(df['Embarked'])
temp1=pd.get_dummies(test['Embarked'])
for x in temp.columns.values:
	df[x]=temp[x]
	test[x]=temp1[x]


for x in drop_columns:
	df.drop(x,axis=1,inplace=True)
	test.drop(x,axis=1,inplace=True)

df['Age']=df['Age'].fillna(df['Age'].mean())
test['Age']=test['Age'].fillna(df['Age'].mean())
test['Fare']=test['Fare'].interpolate()

accuracy=[]
parameters=[]
for p in percentile:
	for c in learning_rate:
		for e in n_estimators:
			for d in max_depth:
				clf=GradientBoostingClassifier(learning_rate=c,n_estimators=e,max_depth=d)
				fs=SelectPercentile(feature_selection.chi2,percentile=p)
				new_tr=fs.fit_transform(df.values,y.values)
				tr1,ts1,tr_l,ts_l=train_test_split(new_tr,y,test_size=.20,random_state=20)
				clf.fit(tr1,tr_l)
				output=clf.predict(ts1)
				print accuracy_score(ts_l,output),p,c,e,d
				accuracy.append(accuracy_score(ts_l,output))
				parameters.append((p,c,e,d))

