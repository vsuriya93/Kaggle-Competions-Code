import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.cross_validation import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.cross_validation import train_test_split

"""

def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              subsample,
              colsample_bytree,
              silent =True,
              nthread = -1,
              seed = 1234):
	clf=xgb.XGBClassifier(max_depth = int(max_depth),
                                         learning_rate = learning_rate,
                                         n_estimators = int(n_estimators),
                                         silent = silent,
                                         nthread = -1,
                                         subsample = subsample,
                                         colsample_bytree = colsample_bytree,
                                         seed = seed,
                                         objective = "binary:logistic")
	xgb_model=clf.fit(meta_train,meta_y,eval_metric="auc",eval_set=[(val_train,val_y)],early_stopping_rounds=25)
	return xgb_model.best_score

"""
df=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

df['Date']=pd.to_datetime(pd.Series(df['Original_Quote_Date']))
df['Year']=df['Date'].apply(lambda x: int(str(x)[:4]))
df['Month']=df['Date'].apply(lambda x: int(str(x)[5:7]))
df['Date']=df['Date'].apply(lambda x: int(str(x)[8:10]))
#df['Field10'].apply(lambda x : int(x.replace(',','')) )

test['Date']=pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test['Year']=test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month']=test['Date'].apply(lambda x: int(str(x)[5:7]))
test['Date']=test['Date'].apply(lambda x: int(str(x)[8:10]))
#test['Field10'].apply(lambda x : int(x.replace(',','')))

label=df['QuoteConversion_Flag']
df.drop('QuoteConversion_Flag',axis=1,inplace=True)
number=test['QuoteNumber']
drop_columns=['Original_Quote_Date','QuoteNumber']
for names in drop_columns:
	df.drop(names,axis=1,inplace=True)
	test.drop(names,axis=1,inplace=True)

df.fillna(0,inplace=True)
test.fillna(0,inplace=True)

for f in df.columns:
	if df[f].dtype=='object':
		print (f)
		lbl=preprocessing.LabelEncoder()
		lbl.fit(list(df[f])+list(test[f]))
		df[f]=lbl.transform(df[f])
		#lbl.fit(list(test[f]))
		test[f]=lbl.transform(test[f])

"""
meta_train,val_train,meta_y,val_y=train_test_split(df,label,test_size=.25,random_state=1510)

xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (int(4),int(8)),
                                      'learning_rate': (0.035, 0.015),
                                      'n_estimators': (int(1500), int(2200)),
                                      'subsample': (0.7, 0.9),
                                      'colsample_bytree' :(0.7, 0.9)
                                     })



xgboostBO.maximize()

print('Final Results')
print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])
"""
