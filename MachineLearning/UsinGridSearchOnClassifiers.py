from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from pprint import pprint
from time import time
import re

train = pd.DataFrame.from_csv('../input/train.tsv', sep='\t', header=0)
test = pd.DataFrame.from_csv("../input/test.tsv", sep='\t', header=0)
index_test = test.index
target = train["tag"]

# print target.shape
classes = set()
tags = []
for x in target:
    # print x
    y = map(int, re.findall(r'\d+', x))
    # print y
    tags.append(y)
    classes.update(y)
int_target = np.array(tags)
print (classes,len(classes))
print (int_target,int_target.shape)

mlb = MultiLabelBinarizer()
target = mlb.fit_transform(int_target)
print (mlb.classes_)
target = pd.DataFrame(target)
print (target.ix[0,:])

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score,make_scorer
from sklearn.pipeline import Pipeline

train_text = train.ix[:,"Seller":"actual_color"]
test_text = test.ix[:,"Seller":"actual_color"]

train_text_columns = train_text.columns
train_text_columns = train_text_columns[1:]

train_text["complete_text"] = train_text["Seller"].map(str) + " "
test_text["complete_text"] = test_text["Seller"].map(str) + " "

# print train_text["complete_text"].head()
for col in train_text_columns:
    # print col,train_text[col].head()
    train_text["complete_text"] = train_text["complete_text"] + train_text[col].map(str) + " "
    test_text["complete_text"] = test_text["complete_text"] + test_text[col].map(str) + " "

print (train_text["complete_text"].head())

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
# print train_text.shape

# print train_text["complete_text"].iloc[2800:2804]
target = target[train_text["complete_text"].notnull()]
train_text = train_text[train_text["complete_text"].notnull()]

test_text.loc[test_text["complete_text"].isnull(),"complete_text"] = ""

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
def remove_nan(line):
    line = line.replace("nan ","")
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    line = pattern.sub('', line)
    # line = WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line))
    return line

text_only = []
test_text_only = []
for idx,x in enumerate(train_text["complete_text"]):
    stripped = strip_tags(x)
    text_only.append(remove_nan(stripped))

for idx, x in enumerate(test_text["complete_text"]):
    stripped = strip_tags(x)
    test_text_only.append(remove_nan(stripped))

text_only = np.array(text_only)
test_text_only = np.array(test_text_only)

print (text_only.shape,target.shape)
print ("text size",test_text_only.shape)

# print ("shape of y: ",target.shape)

# from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from xgboost import XGBClassifier
from sklearn.svm import SVC
pipeline = Pipeline([
    ('vect', CountVectorizer(max_df=0.75,ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=False,norm='l2')),
    ('clf', OneVsRestClassifier(XGBClassifier())),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
# parameters = {
#     'vect__max_df': (0.5, 0.75, 1.0),
#     'vect__max_features': (None, 5000, 10000, 50000),
#     'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
#     'tfidf__use_idf': (True, False),
#     'tfidf__norm': ('l1', 'l2'),
#     'clf__estimator__C': (1,5,10,20),
# }
# XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
#        gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
#        min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
#        objective='multi:softprob', reg_alpha=0, reg_lambda=1,
#        scale_pos_weight=1, seed=0, silent=True, subsample=1)

parameters = {
    'clf__estimator__max_depth': (2, 3, 4),
    'clf__estimator__n_estimators' : (50,100,150,200),
    'clf__estimator__learning_rate' : (0.1,0.2),
    'clf__estimator__min_child_weight' :(1,2),
    'clf__estimator__reg_alpha' :(0,0.1,0.5,1),
    'clf__estimator__reg_lambda' :(0,0.1,0.5,1),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    scorer = make_scorer(f1_score)
    grid_search = GridSearchCV(pipeline, parameters, scoring=scorer,n_jobs=-1,verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    print ("before entering into fit",text_only.shape,target.shape)
    grid_search.fit(text_only, target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # print "accuracy ",accuracy_score(target,clf.predict(train_idf))
    predicted = grid_search.predict(test_text_only)
    print ("after prediction using grid search",predicted, predicted.shape)
    predicted = mlb.inverse_transform(predicted)
    print ("prediction done",predicted)
    print (len(predicted))

    result = []
    for idx, x in enumerate(predicted):
        print
        idx, x
        x = list(x)
        if (len(x) == 0):
            x.append(4483)
        # y = map(int, re.findall(r'\d+', x))
        # print y
        result.append(x)
    result = np.array(result)
    print
    result.shape, result
    to_write = {"item_id": index_test, "tag": result}
    # pd.DataFrame(to_write).to_csv()
    pd.DataFrame(to_write).to_csv("../input/tags_xgb_grid.tsv", index=False, sep="\t")
