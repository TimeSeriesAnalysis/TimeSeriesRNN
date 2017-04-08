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






# Time Series with Echo State Network
+ [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
## Liquid State Machine
+ https://en.wikipedia.org/wiki/Liquid_state_machine
+ http://reservoir-computing.org/software
+ [brian](http://briansimulator.org)
+ `conda install -c brian-team brian2=2.0.1`
`
from brian2 import *
eqs = '''
		dv/dt  = (ge+gi-(v+49*mV))/(20*ms) : volt
		dge/dt = -ge/(5*ms)                : volt
		dgi/dt = -gi/(10*ms)               : volt
	  '''
P = NeuronGroup(4000, eqs, threshold='v>-50*mV', reset='v=-60*mV')
P.v = -60*mV
Pe = P[:3200]
Pi = P[3200:]
Ce = Synapses(Pe, P, on_pre='ge+=1.62*mV')
Ce.connect(p=0.02)
Ci = Synapses(Pi, P, on_pre='gi-=9*mV')
Ci.connect(p=0.02)
M = SpikeMonitor(P)
run(1*second)
plot(M.t/ms, M.i, '.')
show()
`
## Echo State Network
+ [Good introduction](http://www.scholarpedia.org/article/Echo_state_network)
+ [Hierarchical echo-state-machine](http://minds.jacobs-university.de/sites/default/files/uploads/papers/hierarchicalesn_techrep10.pdf)
+ [tutorial](https://www.pdx.edu/sites/www.pdx.edu.sysc/files/Jaeger_TrainingRNNsTutorial.2005.pdf)
+ [book](https://link.springer.com/chapter/10.1007%2F978-3-642-35289-8_36)
+ Other implementations:
	+ https://github.com/sylvchev/simple_esn
	+ http://minds.jacobs-university.de/sites/default/files/uploads/mantas/code/minimalESN_Oger.py.txt
+ [Oger](http://reservoir-computing.org/installing_oger)
+ Install mdp first with conda
+ to install oger download code and `sudo python setup.py install`
+ `import Oger`

## [Extreme Learning Machine](https://en.wikipedia.org/wiki/Extreme_learning_machine)
+ [hierarchical extreme learning machine](http://ieeexplore.ieee.org/document/7280669/)

