import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelBinarizer

df=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

df.fillna(-1,inplace=True)
test.fillna(-1,inplace=True)

def create_mapper(input_values):
#input values should be unique values
	count=0
	mapper={}
	for x in input_values:
		mapper[x]=count
		count=count+1
	return mapper

def my_dummy_function(matrix,input_values,mapper):
# Pass the column values to the pre-allocated matrix, get dummies in the matrix
# the mapping is provided in the mapper dict
	count=0
	for x in input_values:
		val=mapper[x]
		matrix[count][val]=1
		count=count+1
"""
mapper={}
count=0
for x in df.DepartmentDescription.unique():
	mapper[x]=count
	count=count+1

count=0
for x in df.DepartmentDescription:
	val=mapper[x]
	train_mat[count][val]=1
	count=count+1

count=0
for x in test.DepartmentDescription:
	val=mapper[x]
	test_mat[count][val]=1
	count=count+1

day_mapper={}
count=0
for x in df.Weekday:
	day_mapper[x]=count
	count=count+1

df.Weekday.replace(mapper,inplace=True)	
test.Weekday.replace(mapper,inplace=True)	
"""
mapper=create_mapper(df.DepartmentDescription.unique())
train_mat=np.zeros((df.shape[0],len(mapper)))
test_mat=np.zeros((test.shape[0],len(mapper)))
my_dummy_function(train_mat,df.DepartmentDescription,mapper)
my_dummy_function(test_mat,test.DepartmentDescription,mapper)
train_DD=pd.DataFrame(train_mat)
test_DD=pd.DataFrame(test_mat)
df=pd.concat([df,train_DD],axis=1)
test=pd.concat([test,test_DD],axis=1)

day_mapper=create_mapper(df.Weekday.unique())
train_day_matrix=np.zeros((df.shape[0],len(day_mapper)))
test_day_matrix=np.zeros((test.shape[0],len(day_mapper)))
my_dummy_function(train_day_matrix,df.Weekday,day_mapper)
my_dummy_function(test_day_matrix,test.Weekday,day_mapper)
train_DD=pd.DataFrame(train_day_matrix)
test_DD=pd.DataFrame(test_day_matrix)
df=pd.concat([df,train_DD],axis=1)
test=pd.concat([test,test_DD],axis=1)


Upc_mapper=create_mapper(df.Upc.unique())
train_Upc_matrix=np.zeros((df.shape[0],len(Upc_mapper)))
test_Upc_matrix=np.zeros((test.shape[0],len(Upc_mapper)))
my_dummy_function(train_Upc_matrix,df.Upc,day_mapper)
my_dummy_function(test_Upc_matrix,test.Upc,day_mapper)
train_DD=pd.DataFrame(train_Upc_matrix)
test_DD=pd.DataFrame(test_Upc_matrix)
df=pd.concat([df,train_DD],axis=1)
test=pd.concat([test,test_DD],axis=1)

drop=['Weekday','DepartmentDescription']
for x in drop:
	df.drop(x,axis=1,inplace=True)
	test.drop(x,axis=1,inplace=True)

df=pd.pivot_table(df,rows='VisitNumber')
test=pd.pivot_table(test,rows='VisitNumber')
lable=df['TripType']
df.drop('TripType',axis=1,inplace=True)
