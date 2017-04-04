# install all the imported packages

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier   
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn import tree
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold

# training set
df=pd.read_csv('train.csv')
label=df.Class

# test set
test=pd.read_csv('test.csv')
y_true=test.Class

all_entries=pd.read_csv('combined.csv')
all_label=all_entries.Class

# remove unwanted columns
drop_columns=['Bank','Class']
for col_name in drop_columns:
	df.drop(col_name,axis=1,inplace=True)
	test.drop(col_name,axis=1,inplace=True)
	all_entries.drop(col_name,axis=1,inplace=True)

# convert dataframe to array
X=np.array(df)
y=np.array(label)
all_entries=np.array(all_entries)
all_label=np.array(all_label)

pena=['l1','l2']
reg=[.001,.002,.003,.01,.02,.03,.1,.2,.3,1,2,3,4,5,15,30]


number_of_folds=2
kf=KFold(all_entries.shape[0],n_folds=number_of_folds)

max_val=[]
mean_score=[]
mean_across_folds=0
for a in pena:
	for b in reg:
		mean_across_folds=0
		clf=LogisticRegression(penalty=a,C=b)
		for tr,ts in kf:
			clf.fit(all_entries[tr],all_label[tr])
			output=clf.predict(all_entries[ts])
			score=accuracy_score(all_label[ts],output)
			mean_across_folds=mean_across_folds+score
			max_val.append((score,clf))
			print score,a,b,"\t",
		print ' '
		mean_score.append(mean_across_folds/2)

max_index=np.argmax(mean_score)   # has max mean score value

best_model=max_val[max_index * number_of_folds]
print "\n\nBest Score and Model is : ",mean_score[max_index],"\n",best_model[1],"\n"
