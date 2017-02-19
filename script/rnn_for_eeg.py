import numpy as np
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
np.random.seed(0)
#http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
#http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
#http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
# Load dataset

alphabet_size = 30
embedding_vector_length = 200
nb_eeg_sensors = 14
nb_neurons_LSTM = 20
sequence_length=10


EEG_SAX = pd.read_csv("../data/EEG_SAX.csv",header = 0, index_col=0)
X = EEG_SAX[range(14)]
y = pd.DataFrame(EEG_SAX["label"])
size_training_set = 1000
X_train = X.iloc[range(size_training_set)].values
y_train = y.iloc[range(size_training_set)].values
y_train = np.take(y_train,np.asarray(range(sequence_length+1,len(y_train),sequence_length)))
X_train = X_train.reshape(-1,sequence_length,nb_eeg_sensors)
X_train = X_train[0:y_train.shape[0],:,:]

X_test = X.iloc[range(size_training_set,len(X))].values
y_test = y.iloc[range(size_training_set,len(y))].values
y_test = np.take(y_test,np.asarray(range(sequence_length+1,len(y_test),sequence_length)))
X_test = X_test.reshape(-1,sequence_length,nb_eeg_sensors)
X_test = X_test[0:y_test.shape[0],:,:]

print X.columns
print y.columns
# Model creation

model = Sequential()
#model.add(Embedding(alphabet_size, embedding_vector_length, input_length=nb_eeg_sensors))
model.add(LSTM(nb_neurons_LSTM,stateful=True,batch_input_shape=(1,sequence_length,nb_eeg_sensors)))
model.add(Dense(1, activation='relu')) # only one possible outcome: openened or closed
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train,validation_data=(X_test, y_test), nb_epoch=50,  batch_size=1, )
#
# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
