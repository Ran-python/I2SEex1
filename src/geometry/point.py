"""3D point and vector mathematics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple
import numpy as np


@dataclass(frozen=True, slots=True)
class Point3D:
    """Immutable 3D coordinate / vector representation with vector operations."""

    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float) -> None:
        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))
        object.__setattr__(self, "z", float(z))

    @classmethod
    def from_sequence(cls, coords: Sequence[float] | np.ndarray) -> Point3D:
        """Create a Point3D from a 3-element sequence or numpy array."""
        if len(coords) != 3:
            raise ValueError(f"Expected 3 coordinates, got {len(coords)}.")
        return cls(float(coords[0]), float(coords[1]), float(coords[2]))

    def to_numpy(self) -> np.ndarray:
        """Return coordinates as a 1D float64 NumPy array of shape (3,)."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def to_tuple(self) -> Tuple[float, float, float]:
        """Return coordinates as a (x, y, z) float tuple."""
        return (self.x, self.y, self.z)

    def dist(self, other: Point3D) -> float:
        """Compute Euclidean distance between this point and another Point3D."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def norm(self) -> float:
        """Return the Euclidean norm (magnitude) of this vector."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> Point3D:
        """Return a unit vector in the direction of this vector."""
        length = self.norm()
        if length == 0.0:
            return Point3D(0.0, 0.0, 0.0)
        return self / length

    def dot(self, other: Point3D) -> float:
        """Compute the dot product with another 3D vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Point3D) -> Point3D:
        """Compute the cross product with another 3D vector."""
        return Point3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def __add__(self, other: Point3D) -> Point3D:
        if not isinstance(other, Point3D):
            return NotImplemented
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Point3D) -> Point3D:
        if not isinstance(other, Point3D):
            return NotImplemented
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float | int) -> Point3D:
        if not isinstance(scalar, (float, int, np.number)):
            return NotImplemented
        s = float(scalar)
        return Point3D(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, scalar: float | int) -> Point3D:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float | int) -> Point3D:
        if not isinstance(scalar, (float, int, np.number)):
            return NotImplemented
        s = float(scalar)
        if s == 0.0:
            raise ZeroDivisionError("Cannot divide Point3D by zero.")
        return Point3D(self.x / s, self.y / s, self.z / s)

    def __iter__(self) -> Iterable[float]:
        yield self.x
        yield self.y
        yield self.z
