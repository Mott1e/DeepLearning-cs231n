import numpy as np


def l2_regularization(W, reg_strength):
	'''
	Computes L2 regularization loss on weights and its gradient

	Arguments:
		W, np array - weights
		reg_strength - float value

	Returns:
		loss, single value - l2 regularization loss
		gradient, np.array same shape as W - gradient of weight by l2 loss
	'''

	loss = reg_strength * np.sum(W ** 2)
	grad = 2 * reg_strength * W

	return loss, grad


def softmax(predictions):
	'''
	Computes probabilities from scores

	Arguments:
	  predictions, np array, shape is either (N) or (batch_size, N) -
		classifier output

	Returns:
	  probs, np array of the same shape as predictions - 
		probability for every class, 0..1
	'''

	predictions -= np.max(predictions)
	exp_pred = np.exp(predictions)
    
    # probs for (N) shape predictions
	if predictions.shape == (len(predictions), ):
		probs = exp_pred / np.sum(exp_pred)
        
    # probs for (batch_size, N) predictions
	else:
		probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    
	return probs


def cross_entropy_loss(probs, target_index):
	'''
	Computes cross-entropy loss

	Arguments:
	  probs, np array, shape is either (N) or (batch_size, N) -
		probabilities for every class
	  target_index: np array of int, shape is (1) or (batch_size) -
		index of the true class for given sample(s)

	Returns:
	  loss: single value
	'''
    # loss for (N) shape probs
	if probs.shape == (len(probs), ):
		loss = - np.log(probs[target_index])
        
    # loss for (batch_size, N) shape probs
	else:
		loss = -np.log(probs[np.arange(len(probs)), target_index])
		loss = np.mean(loss)
	return loss


def softmax_with_cross_entropy(predictions, target_index):
	'''
	Computes softmax and cross-entropy loss for model predictions,
	including the gradient

	Arguments:
	  predictions, np array, shape is either (N) or (batch_size, N) -
		classifier output
	  target_index: np array of int, shape is (1) or (batch_size) -
		index of the true class for given sample(s)

	Returns:
	  loss, single value - cross-entropy loss
	  dprediction, np array same shape as predictions - gradient of predictions by loss value
	'''

	preds = predictions.copy()
	probs = softmax(preds)
	loss = cross_entropy_loss(probs, target_index)
	mask = np.zeros_like(probs)

    # mask and dprediction for (N) shape predictions
	if predictions.shape == (len(probs), ):
		mask[target_index] = 1
		dprediction = probs - mask
    # mask and dprediction for (batch_size, N) shape predictions
	else:
		mask[np.arange(len(mask)), target_index] = 1
		dprediction = (probs - mask) / (len(target_index))
	
	return loss, dprediction


class Param:
	"""
	Trainable parameter of the model
	Captures both parameter value and the gradient
	"""
	def __init__(self, value):
			self.value = value
			self.grad = np.zeros_like(value)


class ReLULayer:
	def __init__(self):
			self.X = None

	def forward(self, X):
		# Implemented forward pass
		# Hint: you'll need to save some information about X
		# to use it later in the backward pass
		self.X = X
		result = np.maximum(X, 0)
		return result
	

	def backward(self, d_out):
		"""
		Backward pass

		Arguments:
		d_out, np array (batch_size, num_features) - gradient
				of loss function with respect to output

		Returns:
		d_result: np array (batch_size, num_features) - gradient
			with respect to input
		"""
		# Implemented backward pass
		# Your final implementation shouldn't have any loops

		d_result = (self.X > 0) * d_out
	
		return d_result

	def params(self):
		# ReLU Doesn't have any parameters
		return {}


class FullyConnectedLayer:
	def __init__(self, n_input, n_output):
		self.W = Param(0.001 * np.random.randn(n_input, n_output))
		self.B = Param(0.001 * np.random.randn(1, n_output))
		self.X = None

	def forward(self, X):
		# Implemented forward pass
		# Your final implementation shouldn't have any loops
		self.X = X
		return X @ self.W.value + self.B.value


	def backward(self, d_out):
		"""
		Backward pass
		Computes gradient with respect to input and
		accumulates gradients within self.W and self.B

		Arguments:
		d_out, np array (batch_size, n_output) - gradient
				of loss function with respect to output

		Returns:
		d_result: np array (batch_size, n_input) - gradient
			with respect to input
		"""
		# Implemented backward pass
		# Compute both gradient with respect to input
		# and gradients with respect to W and B
		# Add gradients of W and B to their `grad` attribute

		self.B.grad = np.sum(d_out, axis=0).reshape(self.B.value.shape)
  
		self.W.grad = self.X.T @ d_out
		
		d_input = d_out @ self.W.value.T

		return d_input

	def params(self):
		return {'W': self.W, 'B': self.B}
