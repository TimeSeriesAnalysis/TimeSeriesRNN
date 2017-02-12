# TimeSeriesRNN
Just to a give a look at the application of RNN on time series data.
A very good post: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs//).

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
