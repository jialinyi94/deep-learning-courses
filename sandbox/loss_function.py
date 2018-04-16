import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


def get_data(m):
    np.random.seed(3)
    inputs = np.random.randn(1, m)
    labels = 1 * (np.random.rand(1, m) > .5)
    return inputs, labels


def nn_model(w1, w2, inputs):
    z1 = w1 * inputs
    a1 = np.tanh(z1)
    z2 = w2 * a1
    hats = np.divide(1, 1 + np.exp(-z2))
    return hats


def square_loss(w1, w2):
    inputs, labels = get_data(100)
    hats = nn_model(w1, w2, inputs)
    loss = np.sum(np.square(labels - hats)) / 100
    return loss


def cross_entropy(w1, w2):
    inputs, labels = get_data(100)
    hats = nn_model(w1, w2, inputs)
    loss = np.sum(-np.multiply(labels, np.log(hats)) - np.multiply(1 - labels, np.log(1 - hats))) / 100
    return loss


n = 50
m = 80
w1_list = np.linspace(-5, 5, n)
w2_list = np.linspace(-5, 5, m)

w1_grid, w2_grid = np.meshgrid(w1_list, w2_list)

ce_grid = np.zeros(w1_grid.shape)
sq_grid = np.zeros(w1_grid.shape)

for i in range(m):
    for j in range(len(w1_grid[i])):
        ce_grid[i][j] = cross_entropy(w1_grid[i][j], w2_grid[i][j])
        sq_grid[i][j] = square_loss(w1_grid[i][j], w2_grid[i][j])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Plot a basic wireframe.
ax.plot_wireframe(w1_grid, w2_grid, ce_grid, rstride=10, cstride=10, colors='k')
ax.plot_wireframe(w1_grid, w2_grid, sq_grid, rstride=10, cstride=10)

plt.show()
