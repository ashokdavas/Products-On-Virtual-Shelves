import pandas as pd
from sklearn.feature_selection import SelectKBest,SelectPercentile

#Exhaustive list of feature selection we can apply

dev = [6,9,12,15,18,20,25,30]

test = pd.read_csv("../input/test_data_processed.csv")
train =pd.read_csv("../input/train_data_processed.csv")
target =pd.read_csv("../input/target_data_processed.csv")

# sp_25 = SelectPercentile(percentile=25) #Can be used
# sp_50 = SelectPercentile(percentile=50) #Can be used
kbest = SelectKBest(k=100)
#RFE can also be used but generally very time consuming when feature size is high

kbest.fit(train,target)

train_k = kbest.transform(train)
test_k = kbest.transform(test)
print train_k.shape

filename_train_k = "../input/train_k100" + ".csv"
filename_test_k = "../input/train_k100" + ".csv"

pd.DataFrame(train_k).to_csv(filename_train_k, index=False)
pd.DataFrame(test_k).to_csv(filename_test_k, index=False)

