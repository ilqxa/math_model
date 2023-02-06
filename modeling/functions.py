from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel


class Regression(BaseModel):
    x: npt.NDArray[np.float64]

    @property
    def nobs(self) -> int:
        """Количество наблюдений

        Returns:
            int: Количество наблюдений
        """
        return self.x.shape[0]

    @property
    def nfactors(self) -> int:
        """Количество факторов регрессии

        Returns:
            int: Количество факторов
        """
        return self.x.shape[1]


class LinearFunction(Regression):

    def apply(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Вычисление f(x, b) при заданных b

        Args:
            b (npt.NDArray[np.float64]): Вектор коэффициентов

        Raises:
            Exception: Проверка на корректность значений

        Returns:
            npt.NDArray[np.float64]: Вектор результатов
        """
        if b.ndim != 2 or b.shape[1] != 1: raise Exception('b must be a column-vector')
        if self.nfactors + 1 != b.shape[0]: raise Exception('b-vector must have the same number of rows as X has columns')
        return np.dot(self.x, b)
    
    def b_coefs(self, const: int = 1) -> npt.NDArray[np.float64]:
        """Набор значений при каждом коэффициенте b_i при i от 0 до n

        Args:
            const (int, optional): константное значение при коэффициенте b_0. Defaults to 1.

        Returns:
            npt.NDArray[np.float64]: Матрица значений x_ij для каждого b_i
        """
        b0 = const * np.ones((self.nobs, 1))
        return np.append(b0, self.x, axis=1)