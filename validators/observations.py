import numpy as np
import numpy.typing as npt


ScalarTypes = int | float | np.float64
ArrayTypes = list | tuple | npt.NDArray


def check_pairs(obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if obs.ndim != 2: raise Exception('Data must have 2 dimensions')
    if obs.shape[1] != 2: raise Exception('Point must have 2 coordinates')
    return obs

def check_unique_x(obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if np.unique(obs.T[0], return_counts=True)[-1].max() > 1:
        raise Exception('Each X value must be unique')
    return obs

def check_dim_x(x: ScalarTypes | ArrayTypes | None) -> npt.NDArray[np.float64] | None:
    if x is None: return None
    elif isinstance(x, ScalarTypes): data = [x] # type: ignore
    elif isinstance(x, ArrayTypes): data = x # type: ignore
    else: raise TypeError
    res = np.array(data, dtype=np.float64)
    if res.ndim not in (1, 2): raise ValueError
    return res