#####################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import re

#Read either from file after feature selection processing or before that
test = pd.read_csv("../input/test_data_processed.csv")
train =pd.read_csv("../input/train_data_processed.csv")
target =pd.read_csv("../input/target_data_processed.csv")

print("before submitting to tensorflow for result",train.shape,test.shape,target.shape)

print ("divide into test and train for accuracy validation")
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(train,target,test_size=0.2)
print ("train_test sizes are",train_x.shape,test_x.shape,train_y.shape,test_y.shape)

pd.DataFrame(train_x).to_csv("../input/train_x.csv")
pd.DataFrame(test_x).to_csv("../input/test_x.csv")
pd.DataFrame(train_y).to_csv("../input/train_y.csv")
pd.DataFrame(test_y).to_csv("../input/test_y.csv")
pd.DataFrame(test).to_csv("../input/test_actual.csv")
pd.DataFrame(train).to_csv("../input/train_actual.csv")
pd.DataFrame(target).to_csv("../input/target_actual.csv")
######################################################################################
