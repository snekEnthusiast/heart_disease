from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

heart_disease = fetch_ucirepo(id=45) 
X:pd.DataFrame = heart_disease.data.features 
Y:pd.DataFrame = heart_disease.data.targets 
"""
        name     role         type demographic                                        description  units missing_values
0        age  Feature      Integer         Age                                               None  years             no
1        sex  Feature  Categorical         Sex                                               None   None             no
2         cp  Feature  Categorical        None                                               None   None             no
3   trestbps  Feature      Integer        None  resting blood pressure (on admission to the ho...  mm Hg             no
4       chol  Feature      Integer        None                                  serum cholestoral  mg/dl             no
5        fbs  Feature  Categorical        None                    fasting blood sugar > 120 mg/dl   None             no
6    restecg  Feature  Categorical        None                                               None   None             no
7    thalach  Feature      Integer        None                        maximum heart rate achieved   None             no
8      exang  Feature  Categorical        None                            exercise induced angina   None             no
9    oldpeak  Feature      Integer        None  ST depression induced by exercise relative to ...   None             no
10     slope  Feature  Categorical        None                                               None   None             no
11        ca  Feature      Integer        None  number of major vessels (0-3) colored by flour...   None            yes
12      thal  Feature  Categorical        None                                               None   None            yes

13       num   Target      Integer        None                         diagnosis of heart disease   None             no
"""

#prepare paths
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)
#raw data
X = X.join(Y)
X = X[X['ca'].notna()]
X = X[X['thal'].notna()]
Y = X['num']
X = X.drop('num',axis=1)
l = len(X)

def divide(i):
    return int(i*0.8)

x_train, x_test = X.head(l-divide(l)),X.tail(divide(l))
y_train, y_test = Y.head(l-divide(l)),Y.tail(divide(l))
x_train.to_csv("data/raw/xtrain",index=False)
x_test.to_csv("data/raw/xtest",index=False)
y_train.to_csv("data/raw/ytrain",index=False)
y_test.to_csv("data/raw/ytest",index=False)

print("train size:",len(x_train))

#processed data
def bucket(X,bucket_num,feat_name):
    feat = X[feat_name].to_frame()
    kmeans = KMeans(n_clusters=bucket_num).fit(feat.values)
    X[feat_name] = X[feat_name].apply(lambda x:kmeans.predict([[x]])[0])
    
bucket(X,10,'age')
bucket(X,5,'trestbps')
bucket(X,5,'chol')
x_train, x_test = X.head(l-divide(l)),X.tail(divide(l))
y_train, y_test = Y.head(l-divide(l)),Y.tail(divide(l))
x_train.to_csv("data/processed/xtrain",index=False)
x_test.to_csv("data/processed/xtest",index=False)
y_train.to_csv("data/processed/ytrain",index=False)
y_test.to_csv("data/processed/ytest",index=False)

