from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, validator

from base.values import Value


class Operation(BaseModel, ABC):

    def __add__(self, other: Operation | Value) -> Addition:
        return Addition(terms=[self, other])

    @abstractmethod
    def __str__(self) -> str:
        ...


class Addition(Operation):
    """Арифметическая операция сложения вида t_1+t_2+...+t_n, где
    t - слагаемые (terms)
    """
    terms: list[Operation | Value] = Field(description='слагаемые')
    
    def __str__(self) -> str:
        return '+'.join([t.__str__() for t in self.terms])
    

class Multiplication(Operation):
    """Арифметическая операция умножения вида m_1*m_2*...*m_n, где
    m - множители (multiplers)
    """
    multiplers: list[Operation | Value] = Field(description='множители')

    def __str__(self) -> str:
        return '*'.join([m.__str__() for m in self.multiplers])


class Exponentiation(Operation):
    """Арифметическая операция возведения в степень вида base^power, где
    base - основание
    power - натуральный показатель
    """
    base: Operation | Value = Field(description='основание')
    power: Operation | Value = Field(description='показатель')

    def __str__(self) -> str:
        return self.base.__str__() + '^' + str(self.power)