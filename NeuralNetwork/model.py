import numpy as np
from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
	""" Neural network with two fully connected layers """

	def __init__(self, n_input, n_output, hidden_layer_size, reg):
		"""
		Initializes the neural network

		Arguments:

		n_input, int - dimension of the model input
		n_output, int - number of classes to predict
		hidden_layer_size, int - number of neurons in the hidden layer
		reg, float - L2 regularization strength
		"""
		self.reg = reg
		self.layers = [FullyConnectedLayer(n_input=n_input, n_output=hidden_layer_size), ReLULayer(), FullyConnectedLayer(n_input=hidden_layer_size, n_output=n_output)]


	def compute_loss_and_gradients(self, X, y):
		"""
		Computes total loss and updates parameter gradients
		on a batch of training examples

		Arguments:
		X, np array (batch_size, input_features) - input data
		y, np array of int (batch_size) - classes
		"""
		# Before running forward and backward pass through the model,
		# clear parameter gradients aggregated from the previous pass

		for param in self.params().values():
			param.grad = np.zeros_like(param.value)
   
		input = X.copy()
		for layer in self.layers:
			input = layer.forward(input)
   
		loss, grad = softmax_with_cross_entropy(input, y)

		for layer in reversed(self.layers):
			grad = layer.backward(grad)

		for param in self.params().values():
			l2_loss, l2_grad = l2_regularization(param.value, self.reg)
			loss += l2_loss
			param.grad += l2_grad
   

		# TODO Compute loss and fill param gradients
		# by running forward and backward passes through the model

		# After that, implement l2 regularization on all params
		# Hint: self.params() is useful again!

		return loss


	def predict(self, X):
		"""
		Produces classifier predictions on the set

		Arguments:
			X, np array (test_samples, num_features)

		Returns:
			y_pred, np.array of int (test_samples)
		"""
		# TODO: Implement predict
		# Hint: some of the code of the compute_loss_and_gradients
		# can be reused
		pred = np.zeros(X.shape[0], np.int64)

		input = X.copy()
		for layer in self.layers:
			input = layer.forward(input)
   
		prob = softmax(input)
		pred = np.argmax(prob, axis=1)
  
  
		return pred

	def params(self):
		result = {}
		for i in range(len(self.layers)):
			try:
				result[f"W{i}"] = self.layers[i].params()['W']
				result[f"B{i}"] = self.layers[i].params()['B']
			except(KeyError):
				pass

		return result
