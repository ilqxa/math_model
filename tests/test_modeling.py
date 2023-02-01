import numpy as np

from modeling.interpolation import Lagrange


def test_lagrange_test():
    model = Lagrange(points=np.array([(1., 2.), (2., 3.)]))
    assert model.l(0, np.float64(1)) == 1
    assert model.l(0, np.float64(2)) == 0
    assert model.L(np.float64(1)) == 2.
    assert model.L(np.float64(2)) == 3.
    assert model.L(np.float64(3)) == 4.