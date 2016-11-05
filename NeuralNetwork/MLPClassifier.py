import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import MultiLabelBinarizer
import re
#Read either from file after feature selection processing or before that
test = pd.read_csv("../input/test_data_processed.csv")
train =pd.read_csv("../input/train_data_processed.csv")
target =pd.read_csv("../input/target_data_processed.csv")
index_test = test.index

####################################################
# TARGET CLASSES FORMATION INTO ONE HOT PARTITION

train = pd.DataFrame.from_csv('../input/train.tsv', sep='\t', header=0)
target = train["tag"]

##find iteger classes from target column
classes = set()
tags = []
for x in target:
    y = map(int, re.findall(r'\d+', x))
    tags.append(y)
    classes.update(y)
int_target = np.array(tags)

#Binarize the target classes into one hot partition format
#MultiLabelBinarizer also inverse transform the result.

mlb = MultiLabelBinarizer()
target = mlb.fit_transform(int_target)
print (mlb.classes_)
target = pd.DataFrame(target)
print (target.ix[0,:])

print("before submitting to tensorflow for result",train.shape,test.shape,target.shape)

#######################################################

from sklearn.neural_network import MLPClassifier
print ("starting fitting...")

clf = MLPClassifier(verbose=True,max_iter=10000)
# naive_b = GaussianNB()
# lsvr = SGDClassifier()
# lsvr = XGBClassifier()
# lsvr = RandomForestClassifier()
# clf = OneVsRestClassifier(lsvr)
clf.fit(train,target)
print "predictions"
print clf.predict(train[:5,:])

# print "accuracy ",accuracy_score(target,clf.predict(train_idf))
predicted = clf.predict(test)
print ("binary prediction",predicted.shape)
predicted = mlb.inverse_transform(predicted)
# print (predicted)
print ("len predicted",len(predicted))

result = []
for idx,x in enumerate(predicted):
    x = list(x)
    if(len(x) == 0):
        x.append(4483)
    # y = map(int, re.findall(r'\d+', x))
    # print y
    result.append(x)
result = np.array(result)
print result.shape,result
to_write = {"item_id":index_test,"tag":result}
# pd.DataFrame(to_write).to_csv()
pd.DataFrame(to_write).to_csv("../input/mlp.tsv",index=False,sep="\t")
