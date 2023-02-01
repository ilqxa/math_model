import numpy as np
from pydantic import BaseModel, validator


class Lagrange(BaseModel):
    points: np.ndarray

    class Config:
        arbitrary_types_allowed = True
    
    @validator('points')
    def points_must_be_pairs(cls, v: np.ndarray):
        if v.ndim != 2: raise Exception('Data must have 2 dimensions')
        if v.shape[1] != 2: raise Exception('Point must have 2 coordinates')
        return v

    @validator('points')
    def x_must_be_difference(cls, v: np.ndarray):
        if np.unique(v.T[0], return_counts=True)[-1].max() > 1:
            raise Exception('Each X value must be unique')
        return v
        
    @validator('points')
    def points_must_be_float(cls, v: np.ndarray):
        if v.dtype != np.float64: raise TypeError('Points must be defined by float')
        return v
        
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