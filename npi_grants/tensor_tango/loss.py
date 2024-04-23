'''loss function is the same as an error function and is nothing more
than the difference between where I am now and where I want to be'''

import numpy as np
from npi_grants.tensor_tango import tensor

class Loss():
    def loss(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> float:
        """value of loss function is difference of predictions and labels

        Args:
            prediction (tensor.Tensor): predicted values from model
            labels (tensor.Tensor): known data

        Returns:
            float: the value of the loss
        """

        raise NotImplementedError

    def grad(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> tensor.Tensor:
        """gradient of the loss function with respect to predicitons

        Args:
            prediction (tensor.Tensor): predicted values from the model
            labels (tensor.Tensor): known data

        Returns:
            tensor.Tensor: same size as the predictions
        """
        raise NotImplementedError
    
class MSE(Loss):
    def loss(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> float:
        return np.mean((predictions - labels)**2)
    
    def # more