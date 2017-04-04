import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn import metrics

print "\n\nStep1:Loading the data set!!!\n"

df=pd.read_csv('Processed_train')

train=train.drop(train[[0]],1)
train_lable=pd.read_pickle('train_lable_full')
test=pd.read_csv('Processed_test')
test=test.drop(test[[0]],1)
train_lable=np.array(train_lable)

axis=[350]

for num_axis in axis:
	print "Step2:Performing PCA!!\n"
	pca=PCA(n_components=axis,whiten=True)
	pca.fit(train)
	train_final=pca.transform(train)
	clf=LinearSVC(class_weight='auto')
	print "Step3:Fitting SVM!!!\n"
	clf.fit(train_final,train_lable)
	print "Step4:Performing PCA on cross validation set"
	test_final=pca.transform(test)
 	print "Step5:Obtaining the predictions\n"
	output=clf.predict(test_final)

f=open('output.csv','w')
f.write('ID,Target\n')
r=pd.read_pickle('test_id')
ID=np.array(r)
for i in range(len(output)):
	line=str(ID[i])+' '+str(output)+'\n'
	f.write(line)

