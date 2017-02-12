import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import linear

# dimensions: [Batch Size, Sequence Length, Input Dimension]

BATCH_SIZE = None # will be defined latter
SEQUENCE_SIZE = 20 # size of the sequence analyzed 
OBJECT_DIM = 1 # how many features describe one object
NB_CLASSES = 21 # number of classes
NB_HIDDEN_UNITS = 24 # number of units in hidden layer
NB_LAYERS = 3 # number of hidden layers in the neural network
NB_EPOCHS = 50
# input
X = tf.placeholder(tf.float32, [BATCH_SIZE, SEQUENCE_SIZE, OBJECT_DIM])
Y = tf.placeholder(tf.float32, [BATCH_SIZE, NB_CLASSES])

'''Model'''
# Define the initial state of the cells
H_in = tf.placeholder(tf.float32, [None, NB_HIDDEN_UNITS * NB_LAYERS])

# Define the layer that will constitute the RNN 
# Tensorflow defined internally the weights, biases and structure of such a layer
cell = tf.nn.rnn_cell.LSTMCell(NB_HIDDEN_UNITS)
#cell = tf.nn.rnn_cell.GRUCell(NB_HIDDEN_UNITS) # Simpler and fast version of LSTM

# Define the neural network by stacking 3 layers
mcell = tf.nn.rnn_cell.MultiRNNCell([cell]*NB_LAYERS,state_is_tuple=True)

# Internally uses a for loop to "unroll" the RNN 
# the number of times the RNN is unrolled depend on the shape of X
# Consequently it can work with a dataset having sequences with diferent sizes!
# C_state is the last cell state that will be transfered as the input of the layer in the next sequence
# H_r is the outputs of the RNN at each iteration [BATCH_SIZE, SEQUENCE_SIZE, NB_HIDDEN_UNITS]
H_r, C_state = tf.nn.dynamic_rnn(mcell, X, initial_state = H_in, dtype=tf.float32)
H_r_transpose = tf.transpose(H_r, [1, 0, 2])

# Softmax activation layer
Ylogits = linear(H_r_transpose, NB_CLASSES)
Y_nn = tf.nn.softmax(Ylogits)
predictions = tf.argmax(Y_nn, 1)
predictions = tf.reshape(predictions, [BATCH_SIZE,-1])

# loss function and optimizer
loss = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y)
tf.train.AdamOptimizer(1e-3).minimize(loss)

# Accuracy
mistakes = tf.not_equal(Y, Y_nn)
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

# Create session and initialize variables
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

# Training loop
for epoch in range(NB_EPOCHS):
	h_in = np.zeros([BATCH_SIZE, NB_HIDDEN_UNITS * NB_LAYERS])
	for x, y in tf.models.rnn.ptb.reader.ptb_producer(data,BATCH_SIZE,SEQUENCE_SIZE):
		_,y,h_out = sess.run(minimize,{X: x, Y: y, H_in: h_in})
		h_in = h_out
	incorrect = sess.run(error,{X: x, Y: y, H_in: h_in})
	print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()


