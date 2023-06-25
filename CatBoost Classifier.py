# CatBoost Classifier with 1500 iterations and 15 depths
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import make_pipeline, Pipeline
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

import preprocessor
import seaborn as sns

import numpy as np
import pandas as pd

# load data
data=pd.read_csv('_dataa.csv', sep=';')
data = data.dropna()

# Convert data
X = data.drop(['active_inactive'],axis=1)
y = data['active_inactive']

n=5350
a=0;     b=(4*n/5)-1; c=4*n/5 ; d=n-1    # Fold 1
#a=n/5;   b=n-1; c=0 ; d=n/5-1              # Fold 2
#a=2*n/5;  b=n/5-1; c=n/5 ; d=2*n/5-1         # Fold 3              
#a=3*n/5; b=2*n/5-1; c=2*n/5 ; d=3*n/5-1    # Fold 4
#a=4*n/5; b=3*n/5-1; c=3*n/5 ; d=4*n/5-1  # Fold 5

#list1 = list(range(29337,48896))
#list2 = list(range(0,19558))
#list3 = list(range(19558,29337))

#X = X.reindex(list1 + list2 + list3) 
#y = y.reindex(list1 + list2 + list3)

X_train, X_test, y_train, y_test = [], [], [], []
X_train, X_test = X.loc[a:b], X.loc[c:d]
y_train, y_test = y.loc[a:b], y.loc[c:d]

count_inactive_test, count_active_test = 0, 0
for j in y_test:
    if j==0:
        count_inactive_test+=1

    if j==1:
        count_active_test+=1
        
print("inactive firms in the test:", count_inactive_test)
print("active firms in the test:", count_active_test)

# Analyze class imbalance in the targets
# 0 and 1 mean inactive, active firms respectively.
counts_1 = y.sum()
counts_0 = len(y) - counts_1

# The weighting for the imlanabce
weight_for_0 = 1 / counts_0
weight_for_1 = 1 / counts_1


#Model with CrossEntropy
#model = CatBoostClassifier(class_weights=[weight_for_0,weight_for_1], iterations=1500, learning_rate=0.01, depth=15, random_state=1)
is_cat = (X.dtypes != float)
for feature, feat_is_cat in is_cat.to_dict().items():
    if feat_is_cat:
        X[feature].fillna("NAN", inplace=True)

cat_features_index = np.where(is_cat)[0]
pool = Pool(X, y, cat_features=cat_features_index, feature_names=list(X.columns))

model = CatBoostClassifier(max_depth=15, verbose=True, max_ctr_complexity=1, iterations=100).fit(pool)

model.fit(X, y)


#Model score
print(model.score(X, y))

y_pred=model.predict(X)
print(model.score(X, y))

from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)
