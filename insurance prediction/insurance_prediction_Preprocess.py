import sklearn
import pandas as pd
import numpy
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv('test.csv')
dropColumns=['Id','Hazard']
dummies=['T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9', 'T1_V11','T1_V12','T1_V15','T1_V16','T1_V17','T2_V3','T2_V5','T2_V11','T2_V12','T2_V13']
for name in dummies:
	c=pd.Categorical.from_array(df[name])
	i=c.levels
	df[name]=i.get_indexer(df[name])
"""for name in dummies:
	temp=pd.get_dummies(df[name])
	for x in temp.columns.values:
		df[x]=temp[x]
	df=df.drop(name,1)"""
df.to_csv('newTest1.csv')#,sep='\t')