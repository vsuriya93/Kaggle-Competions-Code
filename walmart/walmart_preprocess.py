import pandas as pd
import xgboost as xgb
import numpy as np

df=pd.read_csv('train.csv')

drop=['Upc','VisitNumber','TripType','Weekday']


df.dropna(inplace=True)
label=df.TripType.groupby(df.VisitNumber).min()
# combine multiple visits to a single entry
day_list=df.Weekday.groupby(df.VisitNumber).min()
#visit_number=pd.DataFrame(day_list.index,columns=['VisitNumber'])
days=pd.DataFrame(pd.DataFrame(day_list).icol(0),columns=['Weekday'])

temp=pd.get_dummies(df.DepartmentDescription)
df=pd.concat([df,temp],axis=1)

df=df.groupby(df.VisitNumber).sum()

df=pd.concat([df,days],axis=1)
temp=pd.get_dummies(df.Weekday)
df=pd.concat([df,temp],axis=1)

for x in drop:
	df.drop(x,axis=1,inplace=True)


y=[]
[(lambda x: y.append(label.irow(x)) ) (x) for x in range(label.shape[0])]

X=np.array(df)
y=np.array(y)
"""
clf=xgb.XGBClassifier(n_estimators=1000,subsample=.85,colsample_bytree=.85,learning_rate=.02,max_depth=7)

clf.fit(X,y)
"""
