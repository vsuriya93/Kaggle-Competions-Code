# install all the imported packages

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier   
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
import sys

num_classifiers=int(sys.argv[1])
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

# building the model
#parameter=[{'criterion':['gini','entropy'],'max_depth':[5,6,7,8,9]}]
#clf=GridSearchCV(DecisionTreeClassifier(),parameter)
combined=[]
for i in range(num_classifiers):
	clf=DecisionTreeClassifier()
	clf.fit(X,y)
	output=clf.predict(test)
	combined.append(output)

output=np.sum(combined,axis=0)
in_put=[]
[(lambda x: in_put.append(1) if x >10 else in_put.append(0))(x) for x in output]
print accuracy_score(y_true,in_put)
