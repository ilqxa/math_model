from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, validator

from validators.observations import *
    

class ArithmeticProgression(BaseModel):
    """Арифметическая прогрессия.
    Числовая последовательность вида a1, a1+d, a1+2d, ..., a1+(n-1)d, ...
    последовательность чисел (членов прогрессии), в которой каждое число, начиная со второго,
    получается из предыдущего добавлением к нему постоянного числа d (шага, или разности прогрессии):
    """
    a_1: np.float64 = Field(description='первоначальный член')
    d: int = Field(description='разность прогрессии')

    class Config:
        arbitrary_types_allowed = True

    @property
    def seriesType(self) -> str:
        """Тип прогрессии

        Returns:
            str: Наименование типа
        """
        if self.d == 0: return 'стационарная'
        elif self.d > 0: return 'возрастающая'
        elif self.d < 0: return 'убывающая'
        raise Exception

    def get_term(self, n: int) -> np.float64:
        """Член арифметической прогрессии a_n

        Args:
            n (int): Номер искомого члена прогрессии

        Returns:
            np.float64: Значение n-го члена прогрессии
        """
        a_n = self.a_1 + (n - 1) * self.d
        return a_n
    
    def get_terms_sum(self, m: int, n: int = 1) -> np.float64:
        """Сумма членов арифметической прогрессии от n-ого до m-ого
        Метод реализован по теореме

        Args:
            m (int): Номер последнего в сумме члена прогрессии
            n (int, optional): Номер первого в сумме члена прогрессии. Defaults to 1.

        Returns:
            np.float64: Сумма заданных членов
        """
        a_n = self.get_term(n)
        a_m = self.get_term(m)
        S_mn = (a_m + a_n) / 2 * (m - n + 1)
        return S_mn
    
    def get_terms_prod_vector(self, m: int, n: int = 1) -> np.float64:
        """Произведение членов арифметической прогрессии от n-ого до m-ого
        Метод реализован через поэлементно вычисляемый вектор

        Args:
            m (int): Номер последнего в произведении члена прогрессии
            n (int, optional): Номер первого в произведении члена прогрессии. Defaults to 1.

        Returns:
            np.float64: Произведение заданных членов
        """
        T = np.array([self.get_term(k) for k in range(n, m + 1, 1)])
        return T.prod()