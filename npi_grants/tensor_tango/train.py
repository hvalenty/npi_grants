'''create the framework for training our neural network'''

import loss, mlp, optimizer, tensor, data_iterator


def train(neural_net: mlp.MLP,
          features: tensor.Tensor,
          labels: tensor.Tensor,
          epochs: int = 5000,
          iterator = data_iterator.BatchIterator(), # instantiating it
          loss = loss.MSE(),
          optimizer_obj = optimizer.SGD,
          learning_rate: float = 0.05):
    """Train a NN also known as a multilayer perceptron or 
    fully connected feedforward network

    Args:
        neural_net (mlp.MLP): a defined NN
        features (tensor.Tensor): _description_
        labels (tensor.Tensor): _description_
        epochs (int, optional): _description_. Defaults to 5000.
        iterator (_type_, optional): _description_. Defaults to data_iterator.BatchIterator.
        loss (_type_, optional): _description_. Defaults to loss.MSE.
        optimizer (_type_, optional): _description_. Defaults to optimizer.SGD.
        learning_rate (float, optional): _description_. Defaults to 0.05.
    """

    optim = optimizer_obj(neural_net, learning_rate)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in iterator(features, labels):
            predictions = neural_net.forward(batch[0]) # first result is features
            epoch_loss =+ loss.loss(predictions, labels) # batch[1] is labels
            grad = loss.grad(predictions, labels)
            neural_net.backward(grad)
            optim.step()
            neural_net.zero_parameters()
        print(f'Epoch {epoch} has loss {epoch_loss}')


if __name__ == '__main__':
    import numpy as np
    import layer # put imports as local as possible

    # use XOR because linear functions cannot represent XOR

    features = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ])

    # labels are going to be values of true and false
    labels = np.array([
        [1,0], # false is 1, true is 0
        [0,1], # T
        [0,1], # T
        [1,0]  # F
    ])

    neural_net = mlp.MLP([layer.Tanh(2,2),
                          ]) #more
