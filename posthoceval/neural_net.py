import sympy as sp
from sympy import stats

from .utils import as_iterator_of_size


class ReLU(sp.Max):
    def __new__(cls, x, **kwargs):
        return sp.Max(x, 0, **kwargs)


class NeuralNetwork(object):
    def __init__(self, n_features, n_layers, hidden_units=100, activation=ReLU):
        hidden_units = as_iterator_of_size(hidden_units, n_layers, 'units')
        activation = as_iterator_of_size(activation, n_layers, 'activations')
        in_size = n_features
        inputs = sp.symbols('x1:{}'.format(n_features + 1))
        inputs = sp.Matrix(inputs).reshape(1, n_features)
        x = inputs
        for i, (out_size, act) in enumerate(zip(hidden_units, activation)):
            w = sp.symbols('w_l{layer}_1:{n_inputs}_1:{units}'.format(
                units=out_size + 1, n_inputs=in_size + 1, layer=i + 1))
            w = sp.Matrix(w).reshape(in_size, out_size)
            x = (x * w).applyfunc(act)  # z = f(<x,w>)
            in_size = out_size
        self.inputs = inputs
        self.expr = x
