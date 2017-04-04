import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv('train.csv',low_memory=False)
store=pd.read_csv('store.csv')
test=pd.read_csv('test.csv')

store.dropna(axis=1,inplace=True)
store_dict={}
for x in range(len(store)):
	temp=store.irow(x).values
	store_dict[temp[0]]=(temp[1],temp[2],temp[3])

store.drop('Store',axis=1,inplace=True)
new_columns=store.columns.values

#  Adding the columns to the dictionary

for i in range(len(new_columns)):  
	df[new_columns[i]]=df['Store'].apply( lambda x: store_dict[x][i] )
	test[new_columns[i]]=test['Store'].apply( lambda x: store_dict[x][i] )

test_id=test.Id   # This is the Id for the output.csv file
label=df.Sales    # This is what needs to be predicted
test.drop('Id',axis=1,inplace=True)
df.drop('Sales',axis=1,inplace=True)
df.drop('Customers',axis=1,inplace=True)

df['Year']=df['Date'].apply( lambda x: int(str( x[0:4] ) ) )
df['Month']=df['Date'].apply( lambda x: int(str( x[5:7] ) ) )
df['Date']=df['Date'].apply( lambda x: int(str( x[-2:] ) ) )

test['Year']=test['Date'].apply( lambda x: int(str( x[0:4] ) ) )
test['Month']=test['Date'].apply( lambda x: int(str( x[5:7] ) ) )
test['Date']=test['Date'].apply( lambda x: int(str( x[-2:] ) ) )

mapping={ '0':0 , 'a':1 , 'b':2 , 'c':3 , 'd':4}
for x in df.columns.values:
	if df[x].dtype=='object':
		df[x].replace(mapping,inplace=True)
		test[x].replace(mapping,inplace=True)


