import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, validator

from validators.observations import *


class Lagrange(BaseModel):
    points: npt.NDArray[np.float64]

    class Config:
        arbitrary_types_allowed = True
    
    _paired_points = validator('points', allow_reuse=True)(check_pairs)
    _difference_x = validator('points', allow_reuse=True)(check_unique_x)
        
    @property
    def n(self) -> int:
        return self.points.shape[0] - 1
    
    @property
    def x(self) -> np.ndarray:
        return self.points.T[0]
    
    @property
    def y(self) -> np.ndarray:
        return self.points.T[1]
    
    def l(self, i: int, x: np.float64) -> np.float64:
        x_i = self.x[i]
        return np.prod([(x - x_j) / (x_i - x_j) for j, x_j in enumerate(self.x) if j != i], dtype=np.float64)
    
    def L(self, x: np.float64) -> np.float64:
        return np.sum([y * self.l(i, x) for i, y in enumerate(self.y)])