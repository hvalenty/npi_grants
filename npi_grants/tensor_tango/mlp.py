'''
Multilayer perceptron  or beural network or fully connected neural network, [more]
'''

from npi_grants.tensor_tango import tensor, layer


class MLP():
    def __init__(self, layers: list[layer.Layers]):
        self.layers = layers

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        "compute forward pass through entire network"
        for layer in self.layers:
            x = layer.forward(x)
            return x
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        # more
        return 