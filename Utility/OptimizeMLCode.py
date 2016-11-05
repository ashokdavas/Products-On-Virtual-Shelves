import pandas as pd
import numpy as np
import re

# print un_op_result.head()

train = pd.DataFrame.from_csv('../input/train.tsv', sep='\t', header=0)
test = pd.DataFrame.from_csv("../input/test.tsv", sep='\t', header=0)
index_test = test.index

feature_training = train.ix[:, ["Recommended Use", "Recommended Location", "Publisher", "MPAA Rating", "Literary Genre",
                                "Item Class ID", "ISBN", "Genre ID", "Aspect Ratio", "Actual Color", "Seller"]]
feature_test = test.ix[:, ["Recommended Use", "Recommended Location", "Publisher", "MPAA Rating", "Literary Genre",
                           "Item Class ID", "ISBN", "Genre ID", "Aspect Ratio", "Actual Color", "Seller"]]

# columns = feature_training.columns

column = "Artist ID"
find_indexes = test.loc[test[column].notnull(), :].index
un_op_result = pd.DataFrame.from_csv("../input/86.tsv", sep='\t', header=0)
v =[5065]
for ind in find_indexes:
    print "index and col", ind
    tag_in_ml_data = un_op_result.loc[ind,]
    print "tag_in_ml_data before removing y at above index ", tag_in_ml_data
    y = map(int, re.findall(r'\d+', str(tag_in_ml_data[0])))
    print ("tag in ml ", y)

    tag_in_ml_data = list(set().union([], v))
    print "after adding: tag_in_ml_data ", tag_in_ml_data
    un_op_result.loc[ind, "tag"] = tag_in_ml_data
    filename = "../input/"+"5065"+"_added.tsv"
    to_write = {"item_id":index_test,"tag":np.array(un_op_result["tag"])}
    pd.DataFrame(to_write).to_csv(filename,sep="\t",index=False)


# for col in columns:
#     un_op_result = pd.DataFrame.from_csv("../input/86.tsv", sep='\t', header=0)
#     unique_values_in_col_train = pd.Series(feature_training[col].values.ravel()).unique()
#     unique_values_in_col_test = pd.Series(feature_test[col].values.ravel()).unique()
#     # print unique_values_in_col_test
#     # print unique_values_in_col_test.shape
#     # print unique_values_in_col_train.shape
#     # print unique_values_in_col_train
#     unique_values_in_col_train = set(unique_values_in_col_train).intersection(set(unique_values_in_col_test))
#     # print len(unique_values_in_col_train)
#     unique_values_in_col_train = [x for x in unique_values_in_col_train if str(x) !='nan']
#     print ("unique values after removing nan: ",unique_values_in_col_train)
#
#     for value in unique_values_in_col_train:
#         # print ("common features",value)
#         corresponding_tags = train.loc[train[col] == value,"tag"]
#         total_rows = corresponding_tags.shape[0]
#         # unique_tags = corresponding_tags.unique()
#         value_count = corresponding_tags.value_counts()
#         unique_tags = value_count.index
#         # print ("value and count : ",value_count)
#         for v in unique_tags:
#             c = value_count[v]
#             ratio = float(c)/total_rows
#             # print ratio,total_rows
#             if(total_rows > 1 and ratio >= 0.5):
#                 # print "added something at: ",col,v
#                 find_indexes = test.loc[test[col] == value,:].index
#                 for ind in find_indexes:
#                     print "index and col",ind
#                     tag_in_ml_data = un_op_result.loc[ind,]
#                     print "tag_in_ml_data before removing y at above index ",tag_in_ml_data
#                     y = map(int, re.findall(r'\d+', str(tag_in_ml_data[0])))
#                     print ("tag in ml ",y)
#                     x = map(int, re.findall(r'\d+', v))
#                     print "tag to add ",x
#                     if(ratio>=0.7):
#                         tag_in_ml_data = list(set().union([], x))
#                     else:
#                         tag_in_ml_data = list(set().union(y,x))
#                     print "after adding: tag_in_ml_data ", tag_in_ml_data
#                     un_op_result.loc[ind,"tag"] = tag_in_ml_data
#
#     filename = "../input/"+col+"_added.tsv"
#     to_write = {"item_id":index_test,"tag":np.array(un_op_result["tag"])}
#     pd.DataFrame(to_write).to_csv(filename,sep="\t",index=False)
#
#         # print ("shapes : ",corresponding_tags.shape,unique_tags.shape)
#         # print ("values : ",corresponding_tags,unique_tags)
#
#
