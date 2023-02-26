from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, validator

from validators.observations import *


class Value(BaseModel, ABC):
    sign: str
    meaning: float | None = None

    def __str__(self) -> str:
        return self.sign


class Constant(Value):
    meaning: float


class Variable(Value):
    ...