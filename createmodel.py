import pandas as pd
import tensorflow as tf
source = "data/raw/"
#data
x_train, x_test = pd.read_csv(source+"xtrain"),pd.read_csv(source+"xtest")
y_train, y_test = pd.read_csv(source+"ytrain"),pd.read_csv(source+"ytest")

x_train = tf.convert_to_tensor(x_train)
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(x_train)
#define
model = tf.keras.models.Sequential([
	normalizer,
	tf.keras.layers.Dense(10, activation='relu'),
	tf.keras.layers.Dense(10, activation='relu'),
	tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam',	
				loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
				metrics=['accuracy',tf.keras.metrics.AUC()])
#train
model.fit(x_train, y_train, epochs=50, batch_size=1,verbose=1)
#test
model.evaluate(x_test,  y_test, verbose=2)
#save
print("save? [name/n]: ",end="")
action = input()
if not action in ["n","N",""]:
	model.save("models/"+action+'.keras')