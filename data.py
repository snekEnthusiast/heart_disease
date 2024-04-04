from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

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
x_train, x_test = X.head(l-l//8),X.tail(l//8)
y_train, y_test = Y.head(l-l//8),Y.tail(l//8)
x_train.to_csv("data/raw/xtrain",index=False)
x_test.to_csv("data/raw/xtest",index=False)
y_train.to_csv("data/raw/ytrain",index=False)
y_test.to_csv("data/raw/ytest",index=False)
print(len(x_train))
print(len(pd.read_csv("data/raw/xtrain")))
#processed data
'''buckets:TODO
def perc_cat(x):
    if x<0.25:
        return 0
    if x<0.5:
        return 1
    if x<0.75:
        return 2
    return 3

X['Percentile Rank'] = X.age.rank(pct = True)
X = X.pop('age')'''



