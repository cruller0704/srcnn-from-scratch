import numpy as np
from collections import OrderedDict
from cnn.layers import Affine
from cnn.layers import BatchNormalization
from cnn.layers import Dropout
from cnn.layers import Relu
from cnn.layers import Sigmoid
from cnn.layers import SoftmaxWithLoss
from cnn.gradient import numerical_gradient


class MultiLayerNetExtend:
    """Extended version of fully-connected multi layer neural network

    Has feture of Weight Decay, Dropout, and Batch Normalizaion

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
    use_dropout : Whether to use Dropout
    dropout_ratio : Dropout ratio
    use_batchnorm : Whether to use Batch Normalization
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu',
                 weight_decay_lambda=0, use_dropout=False, dropout_ratio=0.5,
                 use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
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
            if self.use_batchnorm:
                tmp = 'gamma' + str(idx)
                self.params[tmp] = np.ones(hidden_size_list[idx - 1])
                tmp = 'beta' + str(idx)
                self.params[tmp] = np.zeros(hidden_size_list[idx - 1])
                tmp = 'BatchNorm' + str(idx)
                tmp2 = 'gamma' + str(idx)
                tmp3 = 'beta' + str(idx)
                self.layers[tmp] = BatchNormalization(self.params[tmp2],
                                                      self.params[tmp3])
            tmp = 'Activation_function' + str(idx)
            self.layers[tmp] = activation_layer[activation]()

            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ratio)

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

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if 'Dropout' in key or 'BatchNorm' in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        """Calcurate loss function

        Parameters
        ----------
        x : Input data
        t : Supervised label

        Returns
        -------
        Value of loss function
        """
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5*self.weight_decay_lambda*np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, X, T):
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1:
            T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T)/float(X.shape[0])
        return accuracy

    def numerical_gradient(self, X, T):
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
        loss_W = lambda W: self.loss(X, T, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            tmp = 'W' + str(idx)
            grads[tmp] = numerical_gradient(loss_W, self.params[tmp])
            tmp = 'b' + str(idx)
            grads[tmp] = numerical_gradient(loss_W, self.params[tmp])

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                tmp = 'gamma' + str(idx)
                grads[tmp] = numerical_gradient(loss_W, self.params[tmp])
                tmp = 'beta' + str(idx)
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
        self.loss(x, t, train_flg=True)

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
            tmp = 'W' + str(idx)
            grads[tmp] = (self.layers['Affine' + str(idx)].dW +
                          self.weight_decay_lambda*self.params[tmp])
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                tmp = 'BatchNorm' + str(idx)
                grads['gamma' + str(idx)] = self.layers[tmp].dgamma
                grads['beta' + str(idx)] = self.layers[tmp].dbeta

        return grads
