import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

class preprocess:
	def __init__(self,df,test):
		self.df=df
		self.test=test
		self.df.fillna(-1,inplace=True)
		self.test.fillna(-1,inplace=True)
		self.label=self.df.target
		self.df.drop('target',axis=1,inplace=True)
		self.df.drop('ID',axis=1,inplace=True)
		self.test.drop('ID',axis=1,inplace=True)
		
	def get_column_type(self):
		self.s=df.columns.to_series().groupby(df.dtypes).groups
		self.int_columns=self.s[self.s.keys()[1]]
		self.float_columns=self.s[self.s.keys()[2]]
		self.object_columns=self.s[self.s.keys()[0]]
	def object_columns_preprocess(self):
		mapper_columns=['v22','v47','v52','v56','v79','v112','v113','v125']
		for x in mapper_columns:
			mapper={}
			count=0
			for y in self.df[x].unique():
				mapper[count]=y
			self.df[x].replace(mapper,inplace=True)
			self.test[x].replace(mapper,inplace=True)
			self.df.drop(x,inplace=True,axis=1)
			self.test.drop(x,inplace=True,axis=1)

		for x in self.object_columns:
			if x not in mapper_columns:
				temp=pd.get_dummies(self.df[x])
				self.df=pd.concat([self.df,temp],axis=1)
				temp=pd.get_dummies(self.test[x])
				self.test=pd.concat([self.test,temp],axis=1)
				self.df.drop(x,inplace=True,axis=1)
				self.test.drop(x,inplace=True,axis=1)

	def get_train_test_label(self):
		self.get_column_type()
		self.object_columns_preprocess()
		return np.array(self.df),np.array(self.test),np.array(self.label)

class model:
	def __init__(self,train,label):
		self.train=train
		self.label=label
	def xgb_setting_parameters(self):
		self.num_round=2500
		self.num_round1=1500
		self.num_round2=2000
		self.dtrain=xgb.DMatrix(self.train,self.label)
		self.param={}
		self.param['eval_metrix']='logloss'
		self.param['objective']='binary:logistic'
		self.param['subsample']=.8	
		self.param['colsample_bytree']=.8
		self.param['eta']=.02
		self.param['max_depth']=9
		self.param1={}
		self.param1['eval_metrix']='logloss'
		self.param1['objective']='binary:logistic'
		self.param1['subsample']=.8	
		self.param1['colsample_bytree']=.8
		self.param1['max_depth']=8
		self.param1['eta']=.07
		self.param2={}
		self.param2['eval_metrix']='logloss'
		self.param2['objective']='binary:logistic'
		self.param2['subsample']=.8	
		self.param2['colsample_bytree']=.8
		self.param2['eta']=.91
		self.param2['max_depth']=4
	def train_model(self):
		self.xgb_setting_parameters()
		self.clf1=xgb.train(self.param,self.dtrain,self.num_round)
		self.clf2=xgb.train(self.param1,self.dtrain,self.num_round1)
		self.clf3=xgb.train(self.param2,self.dtrain,self.num_round2)
	def predict(self,test):
		self.output1=self.clf1.predict(xgb.DMatrix(test))
		self.output2=self.clf2.predict(xgb.DMatrix(test))
		self.output3=self.clf3.predict(xgb.DMatrix(test))
		self.output=(self.output1+self.output2+self.output3)/3.0
	def write_to_file(self):
		s=pd.read_csv('sample_submission.csv')
		s.PredictedProb=self.output
		s.to_csv('final.csv',index=False)
df=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
data=preprocess(df,test)
train,test,label=data.get_train_test_label()

clf=model(train,label)
clf.train_model()
clf.predict(test)
clf.write_to_file()

#clf=DecisionTreeClassifier()
#clf.fit(train,label)

