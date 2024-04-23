"""create a mechanism for batching data for our neural network"""

import numpy as np
import tensor
from typing import Iterator

class DataIterator():
    def __call__(self, features: tensor.Tensor, labels: tensor.Tensor) -> Iterator:
        '''Batch a set of features and labels
        
        Args:

        Yields:
        
        '''
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True):

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, features: tensor.Tensor, labels: tensor.Tensor) -> Iterator:
        starts = np.arange(0, len(features), )


# MORE



if __name__ == "__main__":
    class Test():
        def __call__(self, val):
            print(2)