from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, validator

from validators.observations import *


class Expression(BaseModel):
    x: Any

    def some_func(self, x: int) -> str:
        raise NotImplementedError


class Polynomial(Expression):
    """Полином вида P(x) = Σ(a_0 + a_1*x^1 + a_2*x^2 + a_i*x^i)
    https://en.wikipedia.org/wiki/Polynomial
    """
    x: npt.NDArray[np.float64] | None
    a: npt.NDArray[np.float64] | None
    i: npt.NDArray[np.intc] | None

    _matrix_x = validator('x', allow_reuse=True)(check_dim_x)
    #TODO: проверить на соответствие размерностей

    def __str__(self) -> str:
        return 'a'
    
    @staticmethod
    def new_power(degree: int, x: ScalarTypes | None) -> Polynomial:
        """Формирование нового степенного полинома степени degree

        Args:
            degree (int): Номер степени
            x (ScalarTypes | None): Исходное число

        Raises:
            ValueError: _description_

        Returns:
            Polynomial: _description_
        """
        if degree < 0: raise ValueError('Номер степени должен быть больше или равен 0')
        i = np.arange(0, degree, dtype=np.intc)
        new_x = np.full(shape=i, fill_value=x) if x is not None else None
        return Polynomial(x=new_x, a=None, i=i)

    def derivative(order: int) -> Polynomial:
        pass