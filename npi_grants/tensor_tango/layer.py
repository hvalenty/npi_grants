'''a layer is a single set of neurons and it has to be able to 
learn via backpropagation as well as run the network forward/ feed forward
'''

import numpy as np

from npi_grants.tensor_tango import tensor


class layer():
    def __init__(self):
        self.w = tensor.Tensor
        self.b = tensor.Tensor
        self.x = None # can eb thought of as inputs
        self.grad_w = 0
        self.grad_b = 0

    def forward(self, x:tensor.Tensor) -> tensor.Tensor:
        '''compute forward pass of neurons in the layer'''
        raise NotImplementedError
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        '''compute backpropagation of neurons in the layer'''
        raise NotImplementedError
    
class Linear(layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)


#lost it from here
    def forward(self, x: np.ndarray) -> np.ndarray:
        return super().forward(x)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        # math chatter
        self.grad_b = np.sum(grad, axis=0)
        self.grad_w = self.x.T @ grad
        return grad @ self.w.T
    

class Activation(layer):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 f,
                 f_prime):
        """initialize 

        Args:
            input_size (int): _description_
            output_size (int): _description_
            f (_type_): _description_
            f_prime (_type_): _description_
        """
        super().__init__()
        self.w = np.random.randn()

    
    # more def here

    def tanh(x: tensor.Tensor) -> tensor.Tensor:
        return np.tanh(x)
    
    def tanh_prime(x: tensor.Tensor) -> tensor.Tensor:
        y = tanh(x)
        return 1 - y**2