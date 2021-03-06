from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

df=pd.read_csv('test.csv')
#target=df['target']
#fp=open('target','w')
#for x in target:
#	fp.write(str(x)+'\n')
#del df['target']
del df['ID']
col_names=df.columns.values

for x in col_names:
	if df[x].dtype==np.dtype('O'): #categorical variable
		c=pd.Categorical.from_array(df[x])
		i=c.levels
		df[x]=i.get_indexer(df[x])
	else:
		if df[x].isnull().sum!=0:
			df[x]=df[x].interpolate()  #interpolate float and int data types
			df[x]=df[x].fillna(-1)

df.to_csv('Processed_test')
#s=StandardScaler()
#scale=s.fit(df)
#transformed=scale.transform(df)
#transformed=pd.DataFrame(transformed)
#transformed.to_csv('Processed_train')
"""
train,other,train_label,other_label=train_test_split(transformed,target,test_size=.40,random_state=33)

test,val,test_label,val_label=train_test_split(other,other_label,test_size=.50,random_state=33)

training_set=pd.DataFrame(train)
train_label=pd.DataFrame(train_label)
training_set.to_pickle('training_set_Final')
train_label.to_pickle('training_lable_Final')

test_set=pd.DataFrame(test)
test_label=pd.DataFrame(test_label)
test_set.to_pickle('test_set_Final')
test_label.to_pickle('test_lable_Final')

cross_validation_set=pd.DataFrame(val)
cross_validation_label=pd.DataFrame(val_label)
cross_validation_set.to_pickle('val_set_Final')
cross_validation_label.to_pickle('Val_lable_Final')

#df.to_pickle('ModifiedTrain')
"""
"""df=df.drop('target',1);
col_names=['VAR_0207','VAR_0213','VAR_0838','VAR_0214','VAR_0157','VAR_0158','VAR_0205','VAR_0206','VAR_0167','VAR_0177','VAR_0156','VAR_0159','VAR_0209','VAR_0168','VAR_0178','VAR_0166','VAR_0169','VAR_0176','VAR_0179','VAR_0208','VAR_0210','VAR_0211','VAR_0073','VAR_0074'];
#inter_pol=['VAR_0208','VAR_0210','VAR_0211','VAR_0074']
categorize=['VAR_0212']
interp_pol=['VAR_0348','VAR_0527','VAR_0529']
problem=['VAR_0008','VAR_0009','VAR_0010','VAR_0011','VAR_0012','VAR_0043','VAR_0157','VAR_0196','VAR_0214','VAR_0225','VAR_0228','VAR_0229','VAR_0231','VAR_0235','VAR_0238']
#g=df.columns.to_series().groupby(df.dtypes).groups
names=df.columns.values

for x in col_names:
	df=df.drop(x,1)

for x in interp_pol:
	df[x]=df[x].interpolate()

for x in names:
	if df[x].isnull().sum()!=0 and x not in col_names:
		df[x]=df[x].interpolate()
for x in names:
	if df[x].isnull().sum!=0 and x not in col_names:
		df=df.drop(x,1)

df.to_pickle('modified_train')

# till here I have no null values, now for the set, you can create the data set

 values to be ignored
 145231: 3 column
 145219:1 column
 144311: 1 column all are NAN
 143142: 1 column looks likt time stamp
 142974: 1 column all NAN 
 142903: 1 column all NAN
 142664: 1 column looks like time stamp
 141873: 1 column looks like time stamp
 139361: 2 columns of time stamp
 135844: 1 colum of all nan
 134506: 1 column of timestamp
 133158: 1 column, timestamp
 131001: 2 columns of timestamp
 127699: 2 columns of time stamp
 125775: 3 columns of floats
 101127: 2 columns 1 is time stamp other is some number
 12550: 1 column, looks like serial number, not unique thouugh 3920 have repeated.
 927: VAR_0348, values are either 1 or 0, see how to make 0 or 1. Interpolation gives .5 and all
 918: there are 286 columns with 918 missing values
 number= number of missing values in each column (1934) elements
 code:

 for i in range(len(number)):
	if number[i]==918:
		print i
 917: 0524: 86 -1 and 144072 as 0
      0527 all are zeros and 144313 are known and a few are unknown
      0529 all are zeros

 91: 56 columns have 91 missing values, same above code, with 91
 
 89: 38 columns have 89 missing values

 60 missing values, df.var_0200, looks like place names, 12386 unique values
 
 56: 115 columns, with missing values, 

 1409: columns with no missing values at all
"""
#df=df.drop('ID',1)
