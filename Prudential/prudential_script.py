import pandas as pd
import numpy as np

df=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

label=df.Response # There are 8 unique values: 1,2,...,8
df.drop('Response',axis=1,inplace=True)

# Roughly replace missing values with -1
df.fillna(-1,inplace=True)
test.fillna(-1,inplace=True)
# Account for the categorical values

mapping={}
unique_values=df.Product_Info_2.unique()

for x in range(len(unique_values)):
	mapping[unique_values[x]]=x

df['Product_Info_2'].replace(mapping,inplace=True)
test['Product_Info_2'].replace(mapping,inplace=True)

drop_col=['Id']
for x in drop_col:
	df.drop(x,axis=1,inplace=True)
	test.drop(x,axis=1,inplace=True)
