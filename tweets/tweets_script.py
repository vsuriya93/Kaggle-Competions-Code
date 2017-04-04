import numpy as np
import xgboost
from sklearn.cross_validation import train_test_split

f=open('output.txt','r')
count=0
feature=[]
for lineno,line in enumerate(f):
	if lineno%2!=0:
		count=count+1
		feature.append(map(float,line.split(' ')[:-1]))

feature=feature[1:]
train=np.array(feature)
f.close()
f=open('label.txt','r')
label=[]
for line in f:
	label.append(line)
print len(label),len(feature)

mapper={}
count=0

for index,x in enumerate(label):
	if x==".\n":
		label[index]='joy\n'

for x in set(label):
	mapper[x]=count
	count=count+1

y=[]
for index,x in enumerate(label):
	y.append(mapper[x])

y=np.array(y)
train=np.array(feature)
X_train,X_test,Y_train,Y_test=train_test_split(train,y,test_size=.15,random_state=22)

