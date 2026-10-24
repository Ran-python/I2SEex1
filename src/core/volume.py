"""3D volumetric raster encapsulation and bounds checking."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

MAX_DIMENSION: int = 1024  # 2^10 per problem specification
SUPPORTED_DTYPES = (np.uint8, np.uint16, np.uint32)


class Volume:
    """Encapsulates a 3D raster matrix M of size sx * sy * sz.

    Attributes:
        data: Contiguous 3D NumPy array of unsigned integers (uint8, uint16, or uint32).
        header: Optional metadata dictionary from the source NRRD file.
    """

    def __init__(
        self,
        data: np.ndarray,
        header: Optional[Dict[str, Any]] = None,
        enforce_dimension_limit: bool = True,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Volume data must be a numpy.ndarray, got {type(data).__name__}.")

        if data.ndim != 3:
            raise ValueError(f"Volume data must be 3-dimensional, got {data.ndim}D array.")

        sx, sy, sz = data.shape
        if sx < 1 or sy < 1 or sz < 1:
            raise ValueError(f"Volume dimensions must be >= 1, got {data.shape}.")

        if enforce_dimension_limit and (
            sx >= MAX_DIMENSION or sy >= MAX_DIMENSION or sz >= MAX_DIMENSION
        ):
            raise ValueError(
                f"Volume dimensions must be strictly less than 2^10 ({MAX_DIMENSION}), "
                f"got ({sx}, {sy}, {sz})."
            )

        if data.dtype not in SUPPORTED_DTYPES:
            raise TypeError(
                f"Volume dtype must be uint8, uint16, or uint32; got {data.dtype}."
            )

        self._data: np.ndarray = np.ascontiguousarray(data)
        self._flat: np.ndarray = self._data.ravel()
        self._header: Dict[str, Any] = dict(header) if header is not None else {}

    @property
    def data(self) -> np.ndarray:
        """Underlying 3D contiguous NumPy array."""
        return self._data

    @property
    def flat_data(self) -> np.ndarray:
        """1D raveled view of the contiguous 3D volume for fast vectorized indexing."""
        return self._flat

    @property
    def header(self) -> Dict[str, Any]:
        """Metadata header dictionary."""
        return self._header

    @property
    def shape(self) -> Tuple[int, int, int]:
        """3D dimensions (sx, sy, sz)."""
        sx, sy, sz = self._data.shape
        return (int(sx), int(sy), int(sz))

    @property
    def sx(self) -> int:
        """Dimension size along X axis."""
        return int(self._data.shape[0])

    @property
    def sy(self) -> int:
        """Dimension size along Y axis."""
        return int(self._data.shape[1])

    @property
    def sz(self) -> int:
        """Dimension size along Z axis."""
        return int(self._data.shape[2])

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of voxel entries (uint8, uint16, or uint32)."""
        return self._data.dtype

    @property
    def bit_depth(self) -> int:
        """Bit depth of each voxel (8, 16, or 32)."""
        return int(self._data.dtype.itemsize * 8)

    def is_in_bounds(self, x: float, y: float, z: float) -> bool:
        """Return True if (x, y, z) lies strictly within [0, sx) x [0, sy) x [0, sz)."""
        sx, sy, sz = self._data.shape
        return 0.0 <= x < sx and 0.0 <= y < sy and 0.0 <= z < sz

    def sample(self, x: float, y: float, z: float) -> int:
        """Sample voxel value M[int(floor(x)), int(floor(y)), int(floor(z))], or 0 if out of bounds."""
        sx, sy, sz = self._data.shape
        if not (0.0 <= x < sx and 0.0 <= y < sy and 0.0 <= z < sz):
            return 0
        return int(self._data[int(x), int(y), int(z)])
