# TimeSeriesRNN

Just to a give a look at the application of RNN on time series data.
A very good post: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs//).

## Some theory regarding LSTM

+ Traditional RNN receive two inputs, the previous output $h_{t-1}$ and the the current data instance $X_t$, then 
using a traditional layer (tanh) it computes the next output $h_t$. The drawbacks of this method are that long term dependencies
are likely to get lost.
+ LSTM NN allow to cope with this problems. In this case the NN will compute the next output using the data instance $X_t$, 
the previous output $h_{t-1}$ and the so called cell-state $C_{t-1}$. 
	+ A sigmoid layer (outputs between 0 and 1) called the "forget gate layer" takes as input $X_t$ and $h_{t-1}$
	and computes which values from the $C_{t-1}$ should be kept or should be erased to compute the next output.
	+ A sigmoid layer called the "input gate layer" layer takes also $X_t$ and $h_{t-1}$ as input and decides
	which values from the $C_{t-1}$ should be updated using the new information.
	+ A first tanh layer (outputs between -1 and 1) takes $X_t$ and $h_{t-1}$ as input and computes the "new candidate values"
	for the cell-state called $C^~$.
	+ The current cell-state $C_{t-1}$ is multiplied by the output of the "forget gate layer" layer in order to keep only 
	the elements of $C_{t-1}$ chosen by the first layer.
	+ The output of the first tanh layer, $C^~$, is multiplied the output of the "input gate layer" and then only the 
	values to be updated are kept.
	+ The two previous outputs are added to produce the new cell-state $C_{t}$
	+ A last sigmoid layer receives $h_{t-1}$ and $X_t$ and outputs wich elements from the new cell-state 
	$C_{t}$ will be output as $h_t$
	+ Finally the cell state $C_t$ passes through a tanh layer (to have elements between -1 and 1) and is multiplied
	by the output of the previous layer

## Some links

+ Nice blogs to start with RNN in tensorflow:
	+ [simple example](http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/)
	+ [another simple implementation](https://gist.github.com/nivwusquorum/b18ce332bde37e156034e5d3f60f8a23)
	+ [another](https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/)
	+ [a less simple example ](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb)
	+ [how to structurate the code! to do after the first example is working ;) ](http://danijar.com/structuring-your-tensorflow-models/)
	+ [extra features](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)
	
## Idea for Time Series analysis like EEG:
+ Variant1:
	+ Use SAX to get a sequence of letters
	+ Une a one-hot-encoding to pass the values to a RNN
	+ 2 models running on 2 separated threads:
		+ Model for training (background)
		+ Model for prediction (take the state of the last trained model)
	+ The output could be:
		+ The value of the next element (regression)
		+ Some class (e.g., epileptic attack, movement ...)
+ Variant 2:
	+ The same but using the mean value instead of a one-hot-encoding
+ Variant 3:
	+ Is it possible to build sequences of patterns instead of sequences of letters only? like using word2vec?
		
## Using Keras
+ http://www.pyimagesearch.com/2016/07/18/installing-keras-for-deep-learning/

## TODO:
+ Make the minimal example work with satic sax EEG data (2sigma data?)
+ Dump and Load a model
+ Open 2 models in 2 threads
+ Make the entire example work: online SAX + RNN
