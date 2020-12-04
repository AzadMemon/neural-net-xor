import numpy as np
import matplotlib.pyplot as plt
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
    learning_rate = None
    max_epochs = None
    error_threshold = None
    error_history = []
    h1 = None # Keeping this to reduce calculations during backprop
    y_pred = None # Keeping this to reduce caluclations during backprop

    '''
    train_X: (n_samples, n_features)
    train_Y: (n_sample, )
    '''

    def __init__(self, train_X, train_Y, learning_rate, error_threshold, max_epochs):
        self.train_X = train_X
        self.train_Y = train_Y
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.error_threshold = error_threshold

    def train(self):
        for i in range(self.max_epochs):
            self._forward_propagation(self.train_X)
            j = self._cost_function(self.y_pred)
            self.error_history.append(j)
            dj_dw1, dj_dw2 = self._backward_propagation()
            self._update_weights(dj_dw1, dj_dw2)

    def predict(self, X):
        y_pred = self._forward_propagation(X)
        y_pred = np.where(self.y_pred > 0.5, 1, 0)
        return y_pred

    def plot_error_rate(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        epochs = np.arange(len(self.error_history))
        ax.plot(epochs, self.error_history)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error')
        ax.set_title('Epoch vs Error')
        fig.savefig('epoch_vs_error.png')

    def _forward_propagation(self, X):
        self.h1 = self._sigmoid(self._a_1(X))
        self.y_pred = self._sigmoid(self._a_2(self.h1)) # I assume the output is a sigmoid as well? But should it be rounded to 1 or 0?
        return self.y_pred

    def _backward_propagation(self):
        dj_dy = (self.train_Y - self.y_pred)
        dy_da2 = self._sigmoid_derivative(self._a_2(self.h1))

        da2_dw2 = self.h1.T # -- End computing derivative w.r.t. w2

        da2_dh1 = self.weights_2.T
        dh1_da1 = self._sigmoid_derivative(self._a_1(self.train_X))
        da1_dw1 = self.train_X.T

        dj_dw2 = (dj_dy * dy_da2) @ da2_dw2
        dj_dw1 = ((da2_dh1 @ (dj_dy * dy_da2)) * dh1_da1) @ da1_dw1

        return dj_dw1, dj_dw2

    def _update_weights(self, dj_dw1, dj_dw2):
        self.weights_1 = self.weights_1 + self.learning_rate * dj_dw1
        self.weights_2 = self.weights_2 + self.learning_rate * dj_dw2

    def _a_1(self, X):
        return np.matmul(self.weights_1, X)

    def _a_2(self, h1):
        return np.matmul(self.weights_2, h1)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-1 * z))

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _cost_function(self, y):
        cost = 0.5 * np.sum((self.train_Y - y) ** 2)
        return cost


if __name__ == '__main__':
    train_X = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])
    train_Y = np.array([0, 1, 1, 0])
    nn = NeuralNetXor(train_X.T, train_Y, learning_rate=0.1, error_threshold=0.1, max_epochs=100000)
    nn.train()
    for i in range(4):
        print(nn.predict(train_X[i, :]))
    nn.plot_error_rate()
