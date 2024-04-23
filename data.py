def main():
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
	import requests
	import pandas as pd
	from pathlib import Path
	import zipfile
	from sklearn.model_selection import train_test_split
	from sklearn.impute import SimpleImputer
	import numpy as np

	#1. download and decompress data
	Path("data").mkdir(exist_ok=True)
	r = requests.get(r"https://archive.ics.uci.edu/static/public/45/heart+disease.zip")
	with open("data/heart_disease.zip","wb") as f:
		f.write(r.content)
		f.close()
	with zipfile.ZipFile("data/heart_disease.zip","r") as zip_ref:
		zip_ref.extractall("data/heart_disease")

	#2. read .data files
	cleveland = pd.read_csv("data/heart_disease/processed.cleveland.data")
	hungary = pd.read_csv("data/heart_disease/processed.hungarian.data")
	switzerland = pd.read_csv("data/heart_disease/processed.switzerland.data")
	va = pd.read_csv("data/heart_disease/processed.va.data")

	#3. unify and process data
	c = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
	cleveland.columns = c
	hungary.columns = c
	switzerland.columns = c
	va.columns = c
		#unify
	data = pd.concat([cleveland,hungary,switzerland,va])
	data.columns = c
	print("data size                   :",data.shape[0])
		#handle missing values
	data = data.map(lambda x: np.nan if x == '?' else x)
	data = data.fillna(np.nan)
	imp = SimpleImputer(missing_values=np.nan, strategy='mean')
	imp.fit(data)
	data = pd.DataFrame(imp.transform(data))
	data.columns = c
		#remove duplicates
	data.drop_duplicates(inplace=True)
	print("size without duplicates     :",data.shape[0])
		#handle different heart diagnoses
	data['num'] = data['num'].apply(lambda x: 0 if x==0 else 1)


	#4. split test-train and save
	X = data.drop('num',axis=1)
	Y = data['num']
	x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=404)
	x_train.to_csv("data/xtrain",index=False)
	x_test.to_csv("data/xtest",index=False)
	y_train.to_csv("data/ytrain",index=False)
	y_test.to_csv("data/ytest",index=False)
	#

if __name__ == '__main__':
	main()