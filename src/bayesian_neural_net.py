from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
import pymc3 as pm
import theano.tensor as T
import theano
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


def create_data(plot=False):
    """
    Create training data
    :return:
    """
    X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
    X = scale(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(X[Y == 0, 0], X[Y == 0, 1], label='Class 0')
        ax.scatter(X[Y == 1, 0], X[Y == 1, 1], color='r', label='Class 1')
        sns.despine(); ax.legend()
        ax.set(xlabel='X', ylabel='Y', title='Classification data set');

        plt.show()

    return X, Y, X_train, X_test, Y_train, Y_test


def create_minibatch(data):
    """
    Use this to yield to generate mini batches of data, so that the model can be ran on big amounts of data
    :param data:
    :return:
    """
    rng = np.random.RandomState(0)
    while True:
        ixs = rng.randint(len(data), size=50)
        yield data[ixs]


def train_model(X, X_train, Y_train, n_hidden=10, plot=False):
    """
    Train the Bayesian neural net. It has a single hidden layer and n_hidden nodes
    Train it using minibatches to allow for big amounts of data to be processed
    :param X:
    :param X_train:
    :param Y_train:
    :param n_hidden: Number of nodes for hidden layer
    :param plot: Whether or not to plot
    :return:
    """
    ann_input = theano.shared(X_train)
    ann_output = theano.shared(Y_train)

    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[1], n_hidden)
    init_2 = np.random.randn(n_hidden, n_hidden)
    init_out = np.random.randn(n_hidden)

    with pm.Model() as neural_network:
        # Weights from input to hidden layer
        weights_in_1 = pm.Normal('w_in_1', 0, sd=1,
                                 shape=(X.shape[1], n_hidden),
                                 testval=init_1)

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal('w_1_2', 0, sd=1,
                                shape=(n_hidden, n_hidden),
                                testval=init_2)

        # Weights from hidden layer to output
        weights_2_out = pm.Normal('w_2_out', 0, sd=1,
                                  shape=(n_hidden,),
                                  testval=init_out)

        # Build neural-network using tanh activation function
        act_1 = T.tanh(T.dot(ann_input,
                             weights_in_1))
        act_2 = T.tanh(T.dot(act_1,
                             weights_1_2))
        act_out = T.nnet.sigmoid(T.dot(act_2,
                                       weights_2_out))

        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli('out',
                           act_out,
                           observed=ann_output)

    # Set back to original data to retrain
    ann_input.set_value(X_train)
    ann_output.set_value(Y_train)

    # Tensors and RV that will be using mini-batches
    minibatch_tensors = [ann_input, ann_output]
    minibatch_RVs = [out]

    minibatches = zip(
        create_minibatch(X_train),
        create_minibatch(Y_train),
    )

    total_size = len(Y_train)

    with neural_network:
        # Run advi_minibatch
        v_params = pm.variational.advi_minibatch(
            n=50000, minibatch_tensors=minibatch_tensors,
            minibatch_RVs=minibatch_RVs, minibatches=minibatches,
            total_size=total_size, learning_rate=1e-2, epsilon=1.0
        )

    with neural_network:
        trace = pm.variational.sample_vp(v_params, draws=5000)

    if plot:
        plt.plot(v_params.elbo_vals)
        plt.ylabel('ELBO')
        plt.xlabel('iteration')
        plt.show()

    return ann_input, ann_output, trace, neural_network


def test_model(ann_input, ann_output, trace, neural_network, X_test, Y_test, plot=False):
    """
    Test the accuracy of the model trained
    :param ann_input:
    :param ann_output:
    :param trace:
    :param neural_network:
    :param plot:
    :return:
    """
    # Replace shared variables with testing set
    ann_input.set_value(X_test)
    ann_output.set_value(Y_test)

    # Creater posterior predictive samples
    ppc = pm.sample_ppc(trace, model=neural_network, samples=500)

    # Use probability of > 0.5 to assume prediction of class 1
    pred = ppc['out'].mean(axis=0) > 0.5

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(X_test[pred == 0, 0], X_test[pred == 0, 1])
        ax.scatter(X_test[pred == 1, 0], X_test[pred == 1, 1], color='r')
        sns.despine()
        ax.set(title='Predicted labels in testing set', xlabel='X', ylabel='Y')
        plt.show()

    print('Accuracy = {}%'.format((Y_test == pred).mean() * 100))

    return pred


if __name__ == "__main__":
    X, Y, X_train, X_test, Y_train, Y_test = create_data()
    ann_input, ann_output, trace, neural_network = train_model(X, X_train, Y_train)
    pred = test_model(ann_input, ann_output, trace, neural_network, X_test, Y_test)
