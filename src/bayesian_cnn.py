import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm
from scipy.stats import mode, chisquare
from sklearn.metrics import confusion_matrix, accuracy_score
import lasagne
from utils import load_dataset
import theano
from utils import run_advi

sns.set_style('white')
sns.set_context('talk')

"""
In this file we build the Bayesian Convolutional NN to be trained on the MNIST data set
"""


class GaussWeights(object):
    """
    The priors act as regularizers here to try and keep the weights of the ANN small.
    It’s mathematically equivalent to putting a L2 loss term that penalizes large weights into the objective function
    """
    def __init__(self):
        self.count = 0

    def __call__(self, shape):
        self.count += 1
        return pm.Normal('w%d' % self.count, mu=0, sd=.1,
                         testval=np.random.normal(size=shape).astype(np.float64),
                         shape=shape)


def build_ann_conv(init):
    """
    Build the Convolutional neural net to be trained with the ADVI algorithm
    :param init: can pass in a function init which has to return a Theano expression
    to be used as the weight and bias matrices
    :return:
    """
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.tanh,
        W=init)

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.tanh,
        W=init)

    network = lasagne.layers.MaxPool2DLayer(network,
                                            pool_size=(2, 2))

    n_hid2 = 256
    network = lasagne.layers.DenseLayer(
        network, num_units=n_hid2,
        nonlinearity=lasagne.nonlinearities.tanh,
        b=init,
        W=init
    )

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    network = lasagne.layers.DenseLayer(
        network, num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax,
        b=init,
        W=init
    )

    prediction = lasagne.layers.get_output(network)

    return pm.Categorical('out',
                          prediction,
                          observed=target_var)


if __name__ == "__main__":

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Building a theano.shared variable
    input_var = theano.shared(X_train[:500, ...].astype(np.float64))
    target_var = theano.shared(y_train[:500, ...].astype(np.float64))

    with pm.Model() as neural_network_conv:
        likelihood = build_ann_conv(GaussWeights())
        v_params, trace, ppc, y_pred = run_advi(likelihood, X_train, y_train, input_var, X_test, y_test, target_var)

    print('Accuracy on test data = {}%'.format(accuracy_score(y_test, y_pred) * 100))

    miss_class = np.where(y_test != y_pred)[0]
    corr_class = np.where(y_test == y_pred)[0]

    preds = pd.DataFrame(ppc['out']).T

    chis = preds.apply(lambda x: chisquare(x).statistic, axis='columns')

    sns.distplot(chis.loc[miss_class].dropna(), label='Error')
    sns.distplot(chis.loc[corr_class].dropna(), label='Correct')
    plt.legend()
    sns.despine()
    plt.xlabel('Chi-Square statistic')
    plt.show()
