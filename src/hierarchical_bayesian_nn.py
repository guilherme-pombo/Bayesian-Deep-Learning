import pymc3 as pm
import theano
from sklearn.metrics import confusion_matrix, accuracy_score
import lasagne
from utils import load_dataset
from utils import run_advi
import numpy as np


"""
In this file we build the Hierarchical NN to be trained on the MNIST data set
"""


def build_ann(init):
    """
    Create an ANN with 2 fully connected hidden layers with 800 neurons each
    :param init: can pass in a function init which has to return a Theano expression
    to be used as the weight and bias matrices
    :return:
    """
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    n_hid1 = 800
    l_hid1 = lasagne.layers.DenseLayer(
        l_in, num_units=n_hid1,
        nonlinearity=lasagne.nonlinearities.tanh,
        b=init,
        W=init
    )

    n_hid2 = 800
    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
        l_hid1, num_units=n_hid2,
        nonlinearity=lasagne.nonlinearities.tanh,
        b=init,
        W=init
    )

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
        l_hid2, num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax,
        b=init,
        W=init
    )

    prediction = lasagne.layers.get_output(l_out)

    # 10 discrete output classes -> pymc3 categorical distribution
    out = pm.Categorical('out',
                         prediction,
                         observed=target_var)

    return out


class GaussWeightsHierarchicalRegularization(object):
    """
    The priors act as regularizers here to try and keep the weights of the ANN small.
    It’s mathematically equivalent to putting a L2 loss term that penalizes large weights into the objective function
    """
    def __init__(self):
        self.count = 0

    def __call__(self, shape):
        self.count += 1

        regularization = pm.HalfNormal('reg_hyper%d' % self.count, sd=1)

        return pm.Normal('w%d' % self.count, mu=0, sd=regularization,
                         testval=np.random.normal(size=shape),
                         shape=shape)


if __name__ == "__main__":

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Building a theano.shared variable
    input_var = theano.shared(X_train[:500, ...].astype(np.float64))
    target_var = theano.shared(y_train[:500, ...].astype(np.float64))

    with pm.Model() as neural_network_hier:
        likelihood = build_ann(GaussWeightsHierarchicalRegularization())
        v_params, trace, ppc, y_pred = run_advi(likelihood, X_train, y_train, input_var, X_test, y_test, target_var)

    print('Accuracy on test data = {}%'.format(accuracy_score(y_test, y_pred) * 100))
    pm.traceplot(trace, varnames=['reg_hyper1', 'reg_hyper2', 'reg_hyper3', 'reg_hyper4', 'reg_hyper5', 'reg_hyper6'])

