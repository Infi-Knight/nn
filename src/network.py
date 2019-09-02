import numpy as np
import random


class Network:
    """The biases and weights in the Network object are all initialized randomly,
    using the Numpy `np.random.randn` function to generate Gaussian distributions
    with mean 0 and standard deviation 1.

    Note that the Network initialization code assumes that the first layer of
    neurons is an input layer, and omits to set any biases for those neurons,
    since biases are only ever used in computing the outputs from later layers.

    Parameters:
    ----------
    list_num_neurons:
        contains the number of neurons in the respective layers. So, for
        example, if we want to create a Network object with 2 neurons in the
        first layer, 3 neurons in the second layer, and 1 neuron in the final
        layer, we'd do this with the code:

        ``net = Network([2, 3, 1])``
    """

    def __init__(self, list_num_neurons):
        self.num_layers = len(list_num_neurons)
        self.list_num_neurons = list_num_neurons
        self.biases = [np.random.randn(num_neurons, 1)
                       for num_neurons in list_num_neurons[1:]]
        self.weights = [np.random.randn(num_neurons_next, num_neurons_prev)
                        for num_neurons_prev, num_neurons_next in zip(
            list_num_neurons[:-1], list_num_neurons[1:])]

    def feedforward(self, a):
        """Return the output of the network if "a" is input.
        It is assumed that the input a is an (n, 1) Numpy ndarray,
        not a (n,) vector.
        """
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, lr, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent.

        The code works as follows. In each epoch, it starts by randomly shuffling
        the training data, and then partitions it into mini-batches of the
        appropriate size. This is an easy way of sampling randomly from the
        training data. Then for each mini_batch we apply a single step of
        gradient descent. This is done by the code
        ``self.update_mini_batch(mini_batch, eta)``, which updates the
        network weights and biases according to a single iteration of gradient
        descent, using just the training data in mini_batch.

        Parameters:
        ----------
        training_data:
            The training_data is a list of tuples (x, y) representing the
            training inputs and corresponding desired outputs.
        epochs:
            The number of epochs to train for.
        mini_batch_size:
            Size of the mini-batches to use when sampling.
        lr:
            Learning rate
        test_data:
            If the optional argument test_data is supplied, then the program
            will evaluate the network after each epoch of training, and print
            out partial progress. This is useful for tracking progress, but
            slows things down substantially.
        """
        n_train = len(training_data)
        if test_data:
            n_test = len(test_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(epoch,
                                                    self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(epoch))

    def update_mini_batch(self, mini_batch, lr):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.

        Parameters:
        ----------
        mini_batch:
            A list of tuples ``(x, y)``
        lr:
            The learning rate.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # Del, or nabla - ∇
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropogate(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (lr/len(mini_batch))*nw for w,
                        nw in zip(self.weights, nabla_w)]

        self.biases = [b - (lr/len(mini_batch))*nb for b,
                       nb in zip(self.biases, nabla_b)]

    def backpropogate(self, x, y):
        """Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x.  "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.biases" and "self.weights"."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        #  For each l=2,3,…,L compute zl = wlal−1 + bl and al = σ(zl).
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors layer by layer
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Compute the vector δL = ∇aC ⊙ σ′(zL)
        delta = self.cost_derivative(
            activations[-1], y) * sigmoid_prime(zs[-1])

        # The gradient of the cost function for last layer
        # GRADIENTS: ∂C/∂wljk = al−1k δlj and  ∂C/∂blj = δl
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # backpropogate the error
        # For each l=L−1,L−2,…,2 compute δl = ((wl+1)T δl+1) ⊙ σ′(zl) and the
        # gradient of the cost function (GRADIENTS)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives ∇aC for a
        quadratic cost function"""
        return (output_activations - y)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
