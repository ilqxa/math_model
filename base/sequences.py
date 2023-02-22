from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, validator

from validators.observations import *


class Progression(BaseModel, ABC):
    seq: npt.NDArray[np.float64]

    _vector_x = validator('seq', allow_reuse=True)(check_1dim_x)
    _count_more_one = validator('seq', allow_reuse=True)(check_length)
    _need_common_diff: classmethod

    class Config:
        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def common_difference(self) -> np.float64:
        """Разница между членами прогрессии

        Returns:
            np.float64: Значение разницы
        """

    @staticmethod
    @abstractmethod
    def new_from_init(a_1: np.float64, d: np.float64, n: np.float64) -> Progression:
        """Создание новой прогрессии из данных параметров

        Args:
            a_1 (np.float64): Первый элемент прогрессии
            d (np.float64): Разница между i и i-1 членами
            n (np.float64): Количество членов прогрессии

        Returns:
            Progression: Прогрессия с заданными параметрами
        """

    @property
    def sum(self) -> np.float64:
        """Сумма членов последовательности

        Returns:
            np.float64: Значение суммы
        """
        return self.seq.sum()
    
    @property
    def prod(self) -> np.float64:
        """Произведение членов последовательности

        Returns:
            np.float64: Значение произведения
        """
        return self.seq.prod()

class Arithmetic(Progression):
    _need_common_diff = validator('seq', allow_reuse=True)(check_common_diff)

    @property
    def common_difference(self) -> np.float64:
        return self.seq[1] - self.seq[0]