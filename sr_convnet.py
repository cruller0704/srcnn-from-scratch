import pickle
import numpy as np
from cnn.layers import Convolution
from cnn.layers import Relu
from cnn.layers import MSEAsLoss


class SRConvNet:
    """Super-Resolution Convolutional Neural Network

    Network configuration is as follows.
        Conv - Relu - Conv - Relu - Conv - Loss

    Ref: http://arxiv.org/abs/1501.00092
    """
    def __init__(self, input_dim=(1, 33, 33),
                 conv_param_1={'filter_num': 64, 'filter_size': 9,
                               'pad': 4, 'stride': 1},
                 conv_param_2={'filter_num': 32, 'filter_size': 1,
                               'pad': 0, 'stride': 1},
                 conv_param_3={'filter_num': 1, 'filter_size': 5,
                               'pad': 2, 'stride': 1}):
        # Initialize weights===========
        weight_init_scales = np.array([0.001, 0.001, 0.001])

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2,
                                          conv_param_3]):
            tmp = 'W' + str(idx + 1)
            tmp2 = weight_init_scales[idx]
            self.params[tmp] = tmp2*np.random.randn(conv_param['filter_num'],
                                                    pre_channel_num,
                                                    conv_param['filter_size'],
                                                    conv_param['filter_size'])
            tmp = 'b' + str(idx + 1)
            self.params[tmp] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']

        # Generate layers===========
        self.layers = []
        self.layers.append(Convolution(self.params['W1'],
                                       self.params['b1'],
                                       conv_param_1['stride'],
                                       conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'],
                                       self.params['b2'],
                                       conv_param_2['stride'],
                                       conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W3'],
                                       self.params['b3'],
                                       conv_param_3['stride'],
                                       conv_param_3['pad']))

        self.last_layer = MSEAsLoss()

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):  # PSNR
        y = self.loss(x, t)
        acc = -10*np.log10(y/np.prod(x.shape[1:]) + 1e-7)

        return acc

    def gradient(self, x, t):
        # Forward
        self.loss(x, t)

        # Backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # Set
        grads = {}
        for i, layer_idx in enumerate((0, 2, 4)):
            grads['W' + str(i + 1)] = self.layers[layer_idx].dW
            grads['b' + str(i + 1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name='params.pkl'):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name='params.pkl'):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 4)):
            self.layers[layer_idx].W = self.params['W' + str(i + 1)]
            self.layers[layer_idx].b = self.params['b' + str(i + 1)]
