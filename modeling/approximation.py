from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, validator

from validators.observations import *
from modeling.functions import *


class LeastSquares(BaseModel):
    points: npt.NDArray[np.float64]

    class Config:
        arbitrary_types_allowed = True

    _paired_points = validator('points', allow_reuse=True)(check_pairs)

    def e(self, f: Regression, b: npt.NDArray[np.float64]) -> np.float64:
        """Минимизируемая функция ошибки

        Args:
            b (npt.NDArray[np.float64]): Вектор неизвестных параметров

        Returns:
            np.float64: Совокупность погрешностей
        """
        return np.sum([(y - f(x, b))**2 for x, y in self.points])