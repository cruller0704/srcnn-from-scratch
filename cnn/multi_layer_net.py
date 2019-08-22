import numpy as np
from collections import OrderedDict
from cnn.layers import Affine
from cnn.layers import Relu
from cnn.layers import Sigmoid
from cnn.layers import SoftmaxWithLoss
from cnn.gradient import numerical_gradient


class MultiLayerNet:
    """Fully-connected multi layer neural network

    Parameters
    ----------
    input_size : Input size (784 in the case of MNIST)
    hidden_size_list : List of the number of hidden layer neurons
        (e.g. [100, 100, 100])
    output_size : Output size (10 in the case of MNIST)
    activation : 'relu' or 'sigmoid'
    weight_init_std : Specify the standard deviation of weights (e.g. 0.01)
        He's initial value is set when 'relu' or 'he' is specified
        Xavier' initial value is set when 'sigmoid' or 'xavier' is specified
    weight_decay_lambda : Strength of weight decay (L2-norm)
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu',
                 weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # Initialize weights
        self.__init_weight(weight_init_std)

        # Generate layers
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            tmp = 'Affine' + str(idx)
            self.layers[tmp] = Affine(self.params['W' + str(idx)],
                                      self.params['b' + str(idx)])
            tmp = 'Activation_function' + str(idx)
            self.layers[tmp] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """Set initial weights

        Parameters
        ----------
        weight_init_std : Specify the standard deviation of weights
            (e.g. 0.01)
            He's initial value is set when 'relu' or 'he' is specified
            Xavier' initial value is set when 'sigmoid' or 'xavier' is
            specified
        """
        all_size_list = ([self.input_size] + self.hidden_size_list +
                         [self.output_size])
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2/all_size_list[idx - 1])  # Recommended ...
                # initial value when using ReLU
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1/all_size_list[idx - 1])  # Recommended ...
                # initial value when using sigmoid

            tmp = 'W' + str(idx)
            self.params[tmp] = scale*np.random.randn(all_size_list[idx - 1],
                                                     all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """Calcurate loss function

        Parameters
        ----------
        x : Input data
        t : Supervised label

        Returns
        -------
        Value of loss function
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5*self.weight_decay_lambda*np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """Calculate gradient (numerical derivation)

        Parameters
        ----------
        x : Input data
        t : Supervised label

        Returns
        -------
        Dictionary arguments storing gradient of each layer
            grads['W1'], grads['W2'],... is weights of each layer
            grads['b1'], grads['b2'],... is bias of each layer
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            tmp = 'W' + str(idx)
            grads[tmp] = numerical_gradient(loss_W, self.params[tmp])
            tmp = 'b' + str(idx)
            grads[tmp] = numerical_gradient(loss_W, self.params[tmp])

        return grads

    def gradient(self, x, t):
        """Calculate gradient (backpropagation)

        Parameters
        ----------
        x : Input data
        t : Supervised label

        Returns
        -------
        Dictionary arguments storing gradient of each layer
            grads['W1'], grads['W2'],... is weights of each layer
            grads['b1'], grads['b2'],... is bias of each layer
        """
        # Forward
        self.loss(x, t)

        # Backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Set
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            tmp = 'Affine' + str(idx)
            tmp2 = 'W' + str(idx)
            tmp3 = self.weight_decay_lambda
            grads[tmp2] = self.layers[tmp].dW + tmp3*self.layers[tmp].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
