import numpy as np

from base.sequences import ArithmeticProgression


def test_ap():
    ap = ArithmeticProgression(a_1 = np.float64(1), d = 2)
    
    assert ap.get_term(n=2) == 3
    assert ap.get_term(n=3) == 5
    assert ap.get_term(n=5) == 9

    assert ap.get_terms_sum(m=3, n=1) == 1+3+5
    assert ap.get_terms_sum(m=5, n=3) == 5+7+9
    assert ap.get_terms_sum(m=5, n=1) == 1+3+5+7+9

    assert ap.get_terms_prod_vector(m=3, n=1) == 1*3*5
    assert ap.get_terms_prod_vector(m=5, n=3) == 5*7*9
    assert ap.get_terms_prod_vector(m=5, n=1) == 1*3*5*7*9