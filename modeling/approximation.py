from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, validator

from validators.observations import *


class LeastSquares(BaseModel):
    points: npt.NDArray[np.float64]
    f: Callable[[np.float64, np.ndarray], np.float64]

    class Config:
        arbitrary_types_allowed = True

    _paired_points = validator('points', allow_reuse=True)(check_pairs)

