import numpy as np

def sigmoid (x):
	return 1./(1. + np.exp(-x))  - 0.5     # activation function
def sigmoid_(x):
	return x * (1. - x)             # derivative of sigmoid

np.random.seed(666)
RESERVOIR_SIZE = 1000
INPUT_SIZE = 1
OUTPUT_SIZE = 1
STOCHASTIC_GRADIENT_DECENT_LEARNING_RATE = 0.001
LEAKING_RATE = 0.1
class ESM:
	def __init__(self,
		reservoir_size = RESERVOIR_SIZE,
		input_size = INPUT_SIZE,
		output_size = OUTPUT_SIZE,
		leaking_rate = LEAKING_RATE,
		learning_rate = STOCHASTIC_GRADIENT_DECENT_LEARNING_RATE,
		activation_function_reservoir = sigmoid,
		activation_function_output = sigmoid,
		derivative_activation_function_output = sigmoid_, 
		):
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
		"""
		# Load Parameters
		self.leaking_rate = leaking_rate
		self.learning_rate = learning_rate

		# Echo State Network dimensions 
		self.N = reservoir_size
		self.K = input_size + 1
		self.L = output_size

		# States
		self.u = np.ones(self.K)
		self.x = np.zeros(self.N)
		self.z = np.zeros(self.K + self.N)
		self._pre_y = np.zeros(self.L)
		self.y = np.zeros(self.L)

		# Weights
		self.W = np.random.rand(self.N,self.N) - 0.5
		rhoW = np.max(np.abs(np.linalg.eig(self.W)[0]))
		self.W *= 1.25 / rhoW
		self.W_in = np.random.rand(self.N,self.K) - 0.5
		self.W_fb = 0*(np.random.rand(self.N,self.L) - 0.5)
		self.W_out = np.random.rand(self.L,self.N + self.K)*0.1
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

		delta_x = self.leaking_rate * self.f( self.W_in.dot(self.u) + self.W.dot(self.x) + self.W_fb.dot(self.y) )
		self.x *= (1.-self.leaking_rate)
		self.x += delta_x
		self.z[:self.K] = self.u[:]
		self.z[self.K:] = self.x[:]
		self._pre_y = self.W_out.dot(self.z)
		
		self.y = self.g(self._pre_y)
		if y_target is not None:
			print "o"
			self._stochastic_gradient_decent(y_target)

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

	def fit(self, X, y):
		initLen = 100
		Z = zeros((self.K+self.N,self.X.shape[0]-initLen))
		Y_ESM = y[initLen+1:self.X.shape[0]+1]

		self._update_reservoir(X[0,:])
		for i,x in enumerate(X[1:initLen,:]):
			self._update_reservoir(u = x, y_previous =y[i-1])
		for i,x in enumerate(X[initLen:,:]):
			self._update_reservoir(u = x, y_previous =y[i-1])
			Z[i,:] = self.z[:]
		X_T = X.T
		reg = 0.0000001
		Wout = dot( y.dot(X_T), np.linalg.inv( X.dot(X_T) + reg*np.eye(1+inSize+resSize) ) )
		
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

"""
import matplotlib.pyplot as plt
data = np.loadtxt('MackeyGlass_t17.txt')
#plt.plot(data[0:1000])
#plt.title('A sample of data')
#plt.show()

esm = ESM()
print esm.W_in.shape
print esm.W.shape
print esm.W_in.mean()
print esm.W.mean()


import matplotlib.pyplot as plt 
esm = ESM()

X = np.ones(1000)
X[0:100] *= 0.5
X[100:200] *= 0.7
X[200:300] *= 0.9
X[400:500] *= 0.3
X[500:600] *= 0.1
X[600:700] *= 0.9
X[700:800] *= 0.3

Y = np.sin(np.asarray(range(X.size))*X.T)
X = X.reshape(1000,1)
Y = Y.reshape(1000,1)
plt.plot(range(Y.size)[:1000], Y[:1000])
plt.show()
print Y
print X
quality = []
ans = []
for _ in range(100):
	for i in range(1,100):
		x = X[i,:]
		y = Y[i]
		y_prev = Y[i-1]
		#print x,y
		#esm.predict(x)
		esm.fit(x,y_prev,y)
		quality.append(esm.SSE(y))
		ans.append(esm.y)
	print _
		#ans.append(esm.predict(x))
for i in range(1,1000):
		x = X[i,:]
		y = Y[i-1]
		#print x,y
		esm.predict(x, y)
		#else:
		#	esm.predict(x)
		#ans.append(esm.y)
#plt.ylim(0.4,0.6)
#plt.plot(range(len(ans)), ans)
plt.plot(range(len(quality)), quality)
plt.show()


plt.plot(range(len(ans[:100])), ans[:100])
plt.show()


"""

import pandas as pd
import matplotlib.pyplot as plt

def update_network(a,X,y,W,Wfb,f):
	return (1-a) * X + a * f(W.dot(X) + Wfb.dot(y))


def compute_output(X,Wout):
	return Wout.T.dot(X)


X = np.random.rand(20) - 0.5
W = np.random.rand(20,20) - 0.5
Wbf = (np.random.rand(20,1) - 0.5)
rhoW = np.max(np.abs(np.linalg.eig(W)[0]))
W *= 0.9 / rhoW


X_full = np.ones((200,20))
y = (0.5*np.sin(np.asarray(range(600)) * 0.25)).reshape(600,1)

for i in range(100):
	X = update_network(0.5,X,y[i],W,Wbf,np.tanh)

for i in range(200):
	X_full[i,:] = X
	X = update_network(0.5,X,y[i+100],W,Wbf,np.tanh)


Wout = np.linalg.pinv(X_full).dot(y[:200,:])

plt.plot(pd.DataFrame(X_full))
plt.show()

ans = []
for i in range(300):
	X = update_network(0.5,X,y[i+300-1],W,Wbf,np.tanh)
	ans.append(compute_output(X,Wout))

plt.plot(range(300),y[300:])
plt.plot(range(300),ans)
plt.show()
"""


plt.plot(range(300),y)
plt.show()

for i in range(100):
	X = update_network(0.5,X,W,np.tanh)

"""

