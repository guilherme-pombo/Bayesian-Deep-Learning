from bayesian_neural_net import *
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


print("Training model and making test data predictions...")
X, Y, X_train, X_test, Y_train, Y_test = create_data()
ann_input, ann_output, trace, neural_network = train_model(X, X_train, Y_train)
pred = test_model(ann_input, ann_output, trace, neural_network, X_test, Y_test)


grid = np.mgrid[-3:3:100j, -3:3:100j]
grid_2d = grid.reshape(2, -1).T
dummy_out = np.ones(grid.shape[1], dtype=np.int8)

ann_input.set_value(grid_2d)
ann_output.set_value(dummy_out)

# Creater posterior predictive samples
ppc = pm.sample_ppc(trace, model=neural_network, samples=500)


def plot_probability_surface():
    """
    Plot the probability surface of the test data predictions
    :return:
    """
    cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(*grid, ppc['out'].mean(axis=0).reshape(100, 100), cmap=cmap)
    ax.scatter(X_test[pred == 0, 0], X_test[pred == 0, 1])
    ax.scatter(X_test[pred == 1, 0], X_test[pred == 1, 1], color='r')
    cbar = plt.colorbar(contour, ax=ax)
    _ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='X', ylabel='Y')
    cbar.ax.set_ylabel('Posterior predictive mean probability of class label = 0')

    plt.show()


def plot_uncertainty():
    """
    Plot the standard deviation of the posterior predictive to get a sense for the uncertainty in our predictions
    :return:
    """
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(*grid, ppc['out'].std(axis=0).reshape(100, 100), cmap=cmap)
    ax.scatter(X_test[pred == 0, 0], X_test[pred == 0, 1])
    ax.scatter(X_test[pred == 1, 0], X_test[pred == 1, 1], color='r')
    cbar = plt.colorbar(contour, ax=ax)
    _ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='X', ylabel='Y')
    cbar.ax.set_ylabel('Uncertainty (posterior predictive standard deviation)')

    plt.show()


if __name__ == "__main__":
    plot_uncertainty()
