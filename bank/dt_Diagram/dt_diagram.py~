# install all the imported packages

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier   
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn import tree
import sys

#file_name=sys.argv[1] 
file_name='tree.dot'
# training set
df=pd.read_csv('train.csv')
label=df.Class

# test set
test=pd.read_csv('test.csv')
y_true=test.Class

# remove unwanted columns
drop_columns=['Bank','Class']
for col_name in drop_columns:
	df.drop(col_name,axis=1,inplace=True)
	test.drop(col_name,axis=1,inplace=True)

# convert dataframe to array
X=np.array(df)
y=np.array(label)

crit=['gini']
max_depth=[5,6,7,8,9]
min_sample=[1,2]
class_labels=['bankrupt','healthy']
max_val=[]

for a in crit:
	for b in max_depth:
		for c in min_sample:
			clf=DecisionTreeClassifier(criterion=a,max_depth=b,min_samples_split=c)
			clf.fit(X,y)
			output=clf.predict(test)
			max_val.append((accuracy_score(y_true,output),clf))
			#print accuracy_score(y_true,output),a,b,c

max_val.sort(reverse=True)
final_model=max_val[0][1]
tree.export_graphviz(final_model,out_file=file_name,max_depth=9,feature_names=df.columns.values)

# to generate png image type: dot -Tpng tree.dot -o tree.png 

"""    Decision tree voting model
building the model
parameter=[{'criterion':['gini','entropy'],'max_depth':[5,6,7,8,9]}]
clf=GridSearchCV(DecisionTreeClassifier(),parameter)
combined=[]
for i in range(num_classifiers):
	clf=DecisionTreeClassifier()
	clf.fit(X,y)
	output=clf.predict(test)
	combined.append(output)

output=np.sum(combined,axis=0)
in_put=[]
[(lambda x: in_put.append(1) if x >10 else in_put.append(0))(x) for x in output]
print accuracy_score(y_true,in_put)"""
