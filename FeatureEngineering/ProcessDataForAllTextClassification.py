import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import re

train = pd.DataFrame.from_csv('../input/train.tsv', sep='\t', header=0)
test = pd.DataFrame.from_csv("../input/test.tsv", sep='\t', header=0)

index_test = test.index
target = train["tag"]

####################################################
# TARGET CLASSES FORMATION INTO ONE HOT PARTITION
##find iteger classes from target column
classes = set()
tags = []
for x in target:
    y = map(int, re.findall(r'\d+', x))
    tags.append(y)
    classes.update(y)
int_target = np.array(tags)

#Binarize the target classes into one hot partition format
mlb = MultiLabelBinarizer()
target = mlb.fit_transform(int_target)
print mlb.classes_
target = pd.DataFrame(target)
print target.ix[0,:]

####################################################

####################################################
# Processing of data. Assuming Text Classification for all data.

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

train_text = train.ix[:,"Seller":"actual_color"]  #All Training Data
test_text = test.ix[:,"Seller":"actual_color"]

train_text_columns = train_text.columns
train_text_columns = train_text_columns[1:]

#complete_text is the column for concatenating all columns
train_text["complete_text"] = train_text["Seller"].map(str) + " "
test_text["complete_text"] = test_text["Seller"].map(str) + " "

# print train_text["complete_text"].head()
for col in train_text_columns:
    train_text["complete_text"] = train_text["complete_text"] + train_text[col].map(str) + " "
    test_text["complete_text"] = test_text["complete_text"] + test_text[col].map(str) + " "

print train_text["complete_text"].head()

target = target[train_text["complete_text"].notnull()]
train_text = train_text[train_text["complete_text"].notnull()]
#can't remove rows from test data
test_text.loc[test_text["complete_text"].isnull(),"complete_text"] = ""

# text = train_text["complete_text"]
# text.to_csv("../input/complet_text.csv")

#HTML parser to remove html tags from data
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

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
def remove_nan_stopwords(line):
    line = line.replace("nan ","")
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    line = pattern.sub('', line)
    # line = WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) #will not improve score.
    return line

text_only = []
test_text_only = []
for idx,x in enumerate(train_text["complete_text"]):
    stripped = strip_tags(x)
    text_only.append(remove_nan_stopwords(stripped))

for idx, x in enumerate(test_text["complete_text"]):
    stripped = strip_tags(x)
    test_text_only.append(remove_nan_stopwords(stripped))

text_only = np.array(text_only)
test_text_only = np.array(test_text_only)

print "text size",test_text_only.shape,text_only.shape,target.shape


# Vectorize the data using CountVectorizer and TfidfTransformer
# Attributes provided to CountVectorizer and TfidfTransformer are found using GridSearch
vectorizer = CountVectorizer(max_df=0.75,ngram_range=(1,2))
train_v = vectorizer.fit_transform(text_only)
test_v = vectorizer.transform(test_text_only)
print ("after vactorization",train_v.shape,test_v.shape)

tf_transformer = TfidfTransformer(norm='l2',use_idf=False)
train_idf = tf_transformer.fit_transform(train_v)
test_idf = tf_transformer.transform(test_v)
print ("after tfid",train_idf.shape,test_idf)

# Truncate the number of features. Will reduce score somewhat, but fasten the computation significantly
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=120)
svd.fit(train_idf)
train_idf = svd.transform(train_idf)
test_idf = svd.transform(test_idf)

# Outlier Can be removed and PCA can be applied but reducing score in present case.

# print ("outlier removal")
# from scipy import stats
# deviation = (np.abs(stats.zscore(train_idf)) < 15).all(axis=1)
# train_idf = train_idf[deviation]
# target= target[deviation]
# print train_idf.shape, target.shape

# print ("applying PCA...")
# from sklearn.decomposition import PCA
# sp = PCA(n_components=2)
# sp.fit(train_idf,target)
# train_idf = sp.transform(train_idf)
# test_idf = sp.transform(test_idf)
# print train_idf.shape, test_idf.shape

test = test_idf
train = train_idf
target = target

#Save these to csv files so we don't have to process data again.
pd.DataFrame(test).to_csv("../input/test_data_processed.csv")
pd.DataFrame(train).to_csv("../input/train_data_processed.csv")
pd.DataFrame(target).to_csv("../input/target_data_processed.csv")