import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score
df=pd.read_csv('train.csv')
drop=['PassengerId','Survived','Name','Ticket','Cabin','SibSp','Parch','Embarked']
y=df['Survived']

#Replace categorial sex column with numeric values
c=pd.Categorical.from_array(df['Sex'])
i=c.levels
df['Sex']=i.get_indexer(df['Sex'])

#Fill null values in age column by its mean
df['Age'].fillna(df['Age'].mean(),inplace=True)

#Replace cabin with binarization of attributes and drop NaN before that
cabin=pd.get_dummies(df['Embarked'])
for col_names in cabin.columns.values:
	df[col_names]=cabin[col_names]

#Drop unnecessary columns
for col_names in drop:
	df.drop(col_names,axis=1,inplace=True)
"""
pca=PCA(n_components=7,whiten=False)
train=np.array(df)
pca.fit(train)
new=pca.transform(train)

for x in xrange(0,len(new)):
    if y[x]==0:
        color='red'
    else:
        color='blue'
    plt.scatter(new[x][0],new[x][0],c=color)

plt.show()
"""
train=np.array(df)
perc=[]
val=[]
for x in range(1,101):
	perc.append(x)
	x=x+1

train,test,train_lable,test_lable=train_test_split(train,y,test_size=.20,random_state=36)

for x in perc:
	clf=DTC()
	fs=SelectPercentile(sklearn.feature_selection.chi2,percentile=x)
	new_train=fs.fit_transform(train,train_lable)
 	clf.fit(new_train,train_lable)
	new_test=fs.transform(test)
	output=clf.predict(new_test)
	print accuracy_score(test_lable,output)
	val.append(accuracy_score(test_lable,output))

plt.plot(perc,val)
plt.show()
