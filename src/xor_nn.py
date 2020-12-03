import numpy as np
np.random.seed(42)

'''
n_samples = 2
n_features = 4
'''


class NeuralNetXor():
    weights_1 = np.random.rand(2, 2)
    weights_2 = np.random.rand(1, 2)
    train_X = None
    train_Y = None
    prev_error = np.inf
    learning_rate = None
    error_threshold = None

    '''
    train_X: (n_samples, n_features)
    train_Y: (n_sample, )
    '''

    def __init__(self, train_X, train_Y, learning_rate, error_threshold):
        self.train_X = train_X
        self.train_Y = train_Y
        self.learning_rate = learning_rate
        self.error_threshold = error_threshold

    def train(self):
        j = self.forward_propagation()
        while (j - self.prev_error) > self.error_threshold:
            self.prev_error = j
            self.backward_propagation(j)
            j = self.forward_propagation()

    def forward_propagation(self):
        h1 = self.sigmoid(np.matmul(self.weights_1, self.train_X))
        y = self.sigmoid(np.matmul(self.weights_2, h1))
        j = self.cost_function(y)
        return j

    def backward_propagation(self, j):
        pass

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-1 * z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)(1 - self.sigmoid(z))

    def cost_function(self, y):
        cost = 1 / self.train_Y.shape * np.sum((self.train_Y - y) ** 2)
        return cost


if __name__ == '__main__':
    pass
