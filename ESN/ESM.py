import numpy as np

def sigmoid (x):
	return 1./(1. + np.exp(-x))  - 0.5     # activation function
def sigmoid_(x):
	return x * (1. - x)             # derivative of sigmoid
def tanh_(x):
	return 1 - (np.tanh(x))**2
def tanh(x):
	return np.tanh(x)
def identity(x):
	return x
def identity_(x):
	return 1.

np.random.seed(666)
RESERVOIR_SIZE = 200
INPUT_SIZE = 1
OUTPUT_SIZE = 1
STOCHASTIC_GRADIENT_DECENT_LEARNING_RATE = 0.001
LEAKING_RATE = 0.9
TIME_CONSTANT = 0.06
NOISE_STRENGTH = 0.001
RHO_W = 0.2
class ESM:
	def __init__(self,
		reservoir_size = RESERVOIR_SIZE,
		input_size = INPUT_SIZE,
		output_size = OUTPUT_SIZE,
		leaking_rate = LEAKING_RATE,
		learning_rate = STOCHASTIC_GRADIENT_DECENT_LEARNING_RATE,
		rhoW = RHO_W,
		activation_function_reservoir = tanh,
		activation_function_output = tanh,
		derivative_activation_function_output = tanh_,
		time_constant = TIME_CONSTANT,
		noise_strength = NOISE_STRENGTH,
		bias=False,
		input_to_output=False):
		"""
		:param reservoir_size: Size of the Echo State Network
		:type reservoir_size: int
		:param input_size: Size input vector
		:type input_size: int
		:param output_size: Size output vector
		:type output_size: int
		:param leaking_rate: Reservoir leaking rate
		:type leaking_rate: float
		:param learning_rate: stochastic gradient decent learning rate
		:type learning_rate: float
		:param activation_function_reservoir: activation function for reservoir neurons
		:type activation_function_reservoir: function
		:param activation_function_output: activation function for output neurons
		:type activation_function_output: function
		:param derivative_activation_function_output: derivative of the activation function for output neurons
		:type derivative_activation_function_output: function
		:param time_constant: time constant integration
		:type time_constant: float
		:param noise_strength: noise strength
		:type noise_strength: float
		"""
		# Load Options bias to output and input to output direct connections
		self.bias = bias
		self.input_to_output = input_to_output

		# Load Parameters
		self.leaking_rate = leaking_rate
		self.learning_rate = learning_rate
		self.time_constant = time_constant
		self.noise_strength = noise_strength
		self.rhoW = rhoW

		# Echo State Network dimensions 
		self.N = reservoir_size
		self.K = input_size
		if self.bias:
			self.K += 1
		self.L = output_size
		self.z_size = self.N
		if self.input_to_output:
			self.z_size += self.K
		# States
		self.u = np.ones(self.K)
		self.x = np.zeros(self.N)
		self.z = np.zeros(self.z_size)
		self._pre_y = np.zeros(self.L)
		self.y = np.zeros(self.L)

		# Weights
		self.W = np.random.rand(self.N,self.N) - 0.5
		rhoW = np.max(np.abs(np.linalg.eig(self.W)[0]))
		self.W *= self.rhoW / rhoW
		self.W_in = np.random.rand(self.N,self.K) - 0.5
		self.W_fb = np.random.rand(self.N,self.L) - 0.5
		self.W_out = np.random.rand(self.L,self.z_size)*0.1
		#self._W_out_gradient = np.random.randn(self.L,self.N + self.K)
		
		# Activation functions
		self.f = activation_function_reservoir
		self.g = activation_function_output
		self.g_ = derivative_activation_function_output
		

	def _update_reservoir(self, u = None, y_previous = None, y_target = None):
		"""
		Compute the new states of the neurons located in the reservoir.
		:param u: input vector
		:type u: array
		:param y_target: output answer vector
		:type y_target: array
		"""
		# predictive mode
		if u is not None:
			self.u[:u.size] = u[:] 
		# generative mode
		else:
			self.u[:self.u.size-1] = self.y[:] 
		# With teacher
		if y_previous is not None:
			self.y[:] = y_previous[:]

		delta_x = self.time_constant * self.f( self.W_in.dot(self.u) + self.W.dot(self.x) + self.W_fb.dot(self.y) + np.random.randn(*self.y.shape)*self.noise_strength)
		self.x *= (1.-self.leaking_rate*self.time_constant)
		self.x += delta_x
		if self.input_to_output:
			self.z[:self.K] = self.u[:]
			self.z[self.K:] = self.x[:]
		else:
			self.z[:] = self.x[:]
		print self.z.shape
		print self.W_out.shape
		print 
		self._pre_y = self.W_out.dot(self.z)
		self.y = self.g(self._pre_y)	

	def _compute_gradient(self, y_target):
		"""
		:param y_target: output answer vector
		:type y_target: array
		"""		
		row_term = (self.y - y_target) * self.g_(self.y)
		return np.dot( row_term.reshape(row_term.size,1), self.z.reshape(1,self.z.size)) 

	def _stochastic_gradient_decent(self, y_ans):
		"""
		:param y_ans: output answer vector
		:type y_ans: array
		"""
		#print self._compute_gradient(y_ans)
		#pass
		print self._compute_gradient(y_ans).sum()
		self.W_out -= self.learning_rate * self._compute_gradient(y_ans)

	def fit_online(self, x, y, y_previous):
		"""
		:param x: input vector
		:type x: array		
		:param y: output answer vector
		:type y: array
		:param y_previous: output answer vector previous iteration
		:type y_previous: array
		"""
		self._update_reservoir(u = x, y_previous = y_previous, y_target = y)
		self._stochastic_gradient_decent(y_target)

	def fit(self, X, y, initLen = 100):
		Z = np.zeros((X.shape[0]-initLen,self.z_size))
		Y_ESM = y[initLen+1:X.shape[0]+1]

		self._update_reservoir(X[0,:])
		for i,x in enumerate(X[1:initLen,:]):
			self._update_reservoir(u = x, y_previous =y[i-1])
		for i,x in enumerate(X[initLen:,:]):
			self._update_reservoir(u = x, y_previous =y[i-1])
			Z[i,:] = self.z[:]
		self.Wout = np.linalg.pinv(Z).dot(Y_ESM)
		return Z,y

	def predict(self,X,y):
		Z = np.zeros((X.shape[0],self.z_size))
		Y_predicted = []
		for i,y_i in enumerate(y):
			x = X[i,:]
			self._update_reservoir(u = x, y_previous =y_i)
			Y_predicted.append(self.y[:])
			Z[i,:] = self.z[:]
		for i,x in enumerate(X[y.shape[0]+1:,:]):
			x = X[i,:]
			self._update_reservoir(u = x)
			print i
			Z[i+y.shape[0],:] = self.z[:]
			Y_predicted.append(self.y[:])
		return Z,Y_predicted
		
	def predict_online(self, x=None, y = None):
		"""
		:param x: input vector
		:type x: array
		:param y: output answer vector previous iteration
		:type y: array
		"""
		self._update_reservoir(u = x, y_previous = y)
		return self.y

	def SSE(self,y_target):
		return np.sum(0.5*(self.y - y)**2)



import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("../data/EEG_SAX.csv",index_col=0)
y = (data.label - data.label.mean())*1./data.label.std()
u = data.drop('label', 1)
u = (u-u.mean())*1./u.std()
_ = plt.plot(range(len(u)),u)
_ = plt.plot(range(len(y)),y)

u = u.values
y = y.values
y = y.reshape(y.shape[0],1)
plt.show()

esm = ESM(reservoir_size = 100,
	input_size = 14,
	output_size = 1,
	leaking_rate = 0.99,
	learning_rate = STOCHASTIC_GRADIENT_DECENT_LEARNING_RATE,
	rhoW = 0.2,
	activation_function_reservoir = tanh,
	activation_function_output = identity,
	derivative_activation_function_output = identity_,
	time_constant = 0.06,
	noise_strength = NOISE_STRENGTH)

training_size = 500
X_full,Y_full = esm.fit(u[:training_size,:],y[:training_size+1])
_=plt.plot(pd.DataFrame(X_full))
plt.show()
print X_full.shape

plt.plot(esm.Wout)
plt.show()

X,Y = esm.predict(u[training_size:1000,:],y[training_size+1:1001])

plt.plot(range(len(Y)),Y)
plt.plot(range(len(Y)),y[training_size+1:1001])
plt.show()

