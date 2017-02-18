import numpy
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
numpy.random.seed(0)


# Load dataset
EEG_SAX = pd.read_csv("../data/EEG_SAX.csv",header = 0, index_col=0)
X = EEG_SAX[range(14)]
y = pd.DataFrame(EEG_SAX["label"])
size_training_set = 1000
X_train = X.iloc[range(size_training_set)].values
y_train = y.iloc[range(size_training_set)].values
X_test = X.iloc[range(size_training_set,len(X))].values
y_test = y.iloc[range(size_training_set,len(y))].values
print X.columns
print y.columns
# Model creation
alphabet_size = 30
embedding_vector_length = 200
nb_eeg_sensors = 14
nb_neurons_LSTM = 100
model = Sequential()
model.add(Embedding(alphabet_size, embedding_vector_length, input_length=nb_eeg_sensors))
model.add(LSTM(nb_neurons_LSTM))
model.add(Dense(1, activation='sigmoid')) # only one possible outcome: openened or closed
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=30, )
	
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))