def main(interface=True):
	import pandas as pd
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import roc_auc_score
	import joblib

	#data
	x_train, x_test, y_train, y_test = 0,0,0,0
	source = "data/"
	def load(source):
		x_train, x_test = pd.read_csv(source+"xtrain"),pd.read_csv(source+"xtest")
		y_train, y_test = pd.read_csv(source+"ytrain"),pd.read_csv(source+"ytest")
		return x_train, x_test, y_train, y_test

	try:
		x_train, x_test, y_train, y_test = load(source)
	except:
		#in case the user can't read
		import data
		data.main()
		x_train, x_test, y_train, y_test = load(source)
	
	log_reg = LogisticRegression(random_state=42,max_iter=10000)
	log_reg.fit(x_train,y_train.values.ravel())
	auc = roc_auc_score(log_reg.predict(x_test),y_test)
	print('AUC :',auc)

	if interface:
		inp = input("save model? [(name)/n]")
		if not inp in ['','n','N']:
			#save
			joblib.dump(log_reg, inp+".joblib.pkl")
	else:
		joblib.dump(log_reg,"model.joblib.pkl")


if __name__ == '__main__':
	main()