from base.operations import Addition, Exponentiation
from base.values import Variable, Constant
from differential.operations import Derivative


def test_addition():
    x_1 = Variable(sign='x_1')
    x_2 = Variable(sign='x_2')
    x_3 = Variable(sign='x_3')
    x_1_2 = Addition(terms=[x_1, x_2])
    assert isinstance(x_1_2, Addition)
    assert len(x_1_2.terms) == 2
    assert x_1 in x_1_2.terms
    assert x_2 in x_1_2.terms
    x_1_2_3 = x_1_2 + x_3
    assert isinstance(x_1_2_3, Addition)
    assert len(x_1_2_3.terms) == 2
    assert x_1_2 in x_1_2_3.terms
    assert x_3 in x_1_2_3.terms
    assert x_1_2_3.__str__() == 'x_1+x_2+x_3'

def test_der():
    x = Variable(sign='x')
    n = Constant(meaning=2, sign='2')
    x_n = Exponentiation(base=x, power=n)
    der = Derivative(func=x_n, order=1)