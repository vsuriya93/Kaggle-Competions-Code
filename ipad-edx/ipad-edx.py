import pandas as pd
import numpy as np

df=pd.read_csv('eBayiPadTrain.csv')
test=pd.read_csv('eBayiPadTest.csv')
drop_col=['description','UniqueID']

#get the labels 
label=df.sold
df.drop('sold',axis=1,inplace=True)

for x in drop_col:
	df.drop(x,axis=1,inplace=True)
	test.drop(x,axis=1,inplace=True)

for x in df.columns.values:
	if df[x].dtype==np.object:
		mapper={}
		count=0
		for entry in df[x].unique():
			mapper[entry]=count
			count=count+1
		df[x].replace(mapper,inplace=True)
		test[x].replace(mapper,inplace=True)
