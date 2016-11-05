import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import re
train = pd.DataFrame.from_csv('../input/train.tsv', sep='\t', header=0)
test = pd.DataFrame.from_csv("../input/test.tsv", sep='\t', header=0)
index_test = test.index

#devide feature and text classification columns
feature_training = train.ix[:,["Recommended Use","Recommended Location","Publisher","MPAA Rating","Literary Genre","Item Class ID","ISBN","Genre ID","Aspect Ratio","Actual Color","Seller"]]
feature_test = test.ix[:,["Recommended Use","Recommended Location","Publisher","MPAA Rating","Literary Genre","Item Class ID","ISBN","Genre ID","Aspect Ratio","Actual Color","Seller"]]

text_training = train.ix[:,["Actors","Product Long Description","Product Name","Product Short Description","Short Description","Synopsis"]]
text_test = test.ix[:,["Actors","Product Long Description","Product Name","Product Short Description","Short Description","Synopsis"]]

# print (feature_training.shape,text_training.shape)
# print (feature_training.columns,text_training.columns)
##########


#####get target and transform to multilabel
target = train["tag"]
print target.shape
classes = set()
tags = []
for x in target:
    y = map(int, re.findall(r'\d+', x))
    tags.append(y)
    classes.update(y)
int_target = np.array(tags)

mlb = MultiLabelBinarizer()
target = mlb.fit_transform(int_target)
print mlb.classes_
target = pd.DataFrame(target)
print target.shape
#######

##transform text classification data to form features for classification

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC,SVC

train_text_columns = text_training.columns

text_training["complete_text"] = text_training[train_text_columns[0]].map(str) + " "
text_test["complete_text"] = text_test[train_text_columns[0]].map(str) + " "

train_text_columns = train_text_columns[1:]

# print train_text["complete_text"].head()
for col in train_text_columns:
    # print col,train_text[col].head()
    text_training["complete_text"] = text_training["complete_text"] + text_training[col].map(str) + " "
    text_test["complete_text"] = text_test["complete_text"] + text_test[col].map(str) + " "

print ("after aggregating data: ",text_training["complete_text"].head())

# text = train_text["complete_text"]
# text.to_csv("../input/complet_text.csv",index=False)

from HTMLParser import HTMLParser
class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

#remove null text rows
target = target[text_training["complete_text"].notnull()]
text_training = text_training[text_training["complete_text"].notnull()]
#can't remove rows from test data
text_test.loc[text_test["complete_text"].isnull(),"complete_text"] = ""

def remove_nan(line):
    line = line.replace("nan ","")
    return line

text_train_only = []
test_text_only = []
for idx,x in enumerate(text_training["complete_text"]):
    stripped = strip_tags(x)
    text_train_only.append(remove_nan(stripped))

for idx, x in enumerate(text_test["complete_text"]):
    stripped = strip_tags(x)
    test_text_only.append(remove_nan(stripped))

train_text_only = np.array(text_train_only)
test_text_only = np.array(test_text_only)

print train_text_only[0]
print test_text_only[0]

print ("after parsing and aggregating: ",train_text_only.shape,test_text_only.shape)

vectorizer = CountVectorizer()
train_v = vectorizer.fit_transform(text_train_only)
test_v = vectorizer.transform(test_text_only)
print ("after vactorization ",train_v.shape,test_v.shape)

tf_transformer = TfidfTransformer()
train_idf = tf_transformer.fit_transform(train_v)
test_idf = tf_transformer.transform(test_v)
print ("after tf_transformer ",train_idf.shape,test_idf.shape,target.shape)

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=120)
svd.fit(train_idf)
train_idf = svd.transform(train_idf)
test_idf = svd.transform(test_idf)
print ("at the end of text data normaliztion",train_idf.shape,test_idf.shape,target.shape)
#########

#####transform features from categorial data to numerical data

# print pd.Series(feature_training.ix[:,"Actual Color"].ravel()).unique()
from sklearn.preprocessing import Imputer,StandardScaler,LabelEncoder,OneHotEncoder

#label encoder to change categorical data into numeric data
le = LabelEncoder()
for col in feature_training.columns:
    #replace null values with most frequent values of the column
    feature_training.loc[feature_training[col].isnull(),col] = str(0)
    feature_test.loc[feature_test[col].isnull(),col] = str(0)
    feature_training[col] = le.fit_transform(feature_training[col])
    feature_test[col] = le.fit_transform(feature_test[col])

print ("after labeling",feature_training.shape)
#standard scaling of data
# std_scaler = StandardScaler(with_mean=False)
# std_scaler.fit(feature_training)
# feature_training = std_scaler.transform(feature_training)
# feature_test = std_scaler.transform(feature_test)

feature_training_df = pd.DataFrame(feature_training)
feature_test_df = pd.DataFrame(feature_test)
traing_test = pd.concat([feature_training_df,feature_test_df])


ohe = OneHotEncoder()
ohe.fit(traing_test)
feature_training = ohe.transform(feature_training)
feature_test = ohe.transform(feature_test)
print type(feature_training)
feature_test = pd.SparseDataFrame([ pd.SparseSeries(feature_test[i].toarray().ravel()) for i in np.arange(feature_test.shape[0]) ])
feature_training = pd.SparseDataFrame([ pd.SparseSeries(feature_training[i].toarray().ravel()) for i in np.arange(feature_training.shape[0]) ])
print type(feature_training)

print ("after one hot partition",feature_training.shape,feature_test.shape)

#####################

###### Merge both data frames and apply classification model


text_training_df = pd.DataFrame(train_idf)
text_test_df = pd.DataFrame(test_idf)

feature_training_df = pd.DataFrame(feature_training)
feature_test_df = pd.DataFrame(feature_test)

# feature_training_df = feature_training
# feature_test_df = feature_test


merged_data_train = pd.concat([feature_training_df,text_training_df],axis=1)
merged_data_test = pd.concat([feature_test_df,text_test_df],axis=1)
print ("merged train data ",merged_data_train.shape, merged_data_test.shape)

#################

##
# Save merged_data_train, merged_data_test, target into csv so that we don't need to process data again
##