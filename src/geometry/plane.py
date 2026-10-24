"""3-point (Isosceles Triangle) and 4-point (Isosceles Trapezoid) plane definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple
import numpy as np

from src.geometry.point import Point3D

EPS: float = 0.001


class InvalidPlaneError(ValueError):
    """Raised when 3D points do not form a valid isosceles triangle or trapezoid plane."""


class Plane(ABC):
    """Abstract base class representing a bounded 2D projection region in 3D space.

    Provides a unified interface for both 3-point (isosceles triangle) and
    4-point (isosceles trapezoid) plane definitions so that the projection
    engine operates polymorphically using a_start1, a_start2, b, c, d_ab, and d_bc.
    """

    def __init__(
        self,
        a_start1: Point3D,
        a_start2: Point3D,
        b: Point3D,
        c: Point3D,
        d_ab: float,
        d_bc: float,
    ) -> None:
        self._a_start1 = a_start1
        self._a_start2 = a_start2
        self._b = b
        self._c = c
        self._d_ab = float(d_ab)
        self._d_bc = float(d_bc)

    @property
    def a_start1(self) -> Point3D:
        """Starting top-left apex/corner point."""
        return self._a_start1

    @property
    def a_start2(self) -> Point3D:
        """Starting top-right apex/corner point."""
        return self._a_start2

    @property
    def b(self) -> Point3D:
        """Bottom-left base point."""
        return self._b

    @property
    def c(self) -> Point3D:
        """Bottom-right base point."""
        return self._c

    @property
    def d_ab(self) -> float:
        """Leg length D_ab = dist(a_start1, b)."""
        return self._d_ab

    @property
    def d_bc(self) -> float:
        """Base length D_bc = dist(b, c)."""
        return self._d_bc

    @property
    def v_ab(self) -> Point3D:
        """Normalized leg vector (b - a_start1) / D_ab."""
        return (self._b - self._a_start1) / self._d_ab

    @property
    def v_ac(self) -> Point3D:
        """Normalized leg vector (c - a_start2) / D_ab."""
        return (self._c - self._a_start2) / self._d_ab

    @property
    def output_dimensions(self) -> Tuple[int, int]:
        """Return output 2D matrix dimensions (int(D_ab), int(D_bc))."""
        return (int(self._d_ab), int(self._d_bc))

    @abstractmethod
    def points(self) -> Tuple[Point3D, ...]:
        """Return the defining 3D points of the plane."""

    @staticmethod
    def from_points(points: Sequence[Point3D], eps: float = EPS) -> Plane:
        """Factory method creating a TrianglePlane (3 points) or TrapezoidPlane (4 points)."""
        if len(points) == 3:
            return TrianglePlane(points[0], points[1], points[2], eps=eps)
        if len(points) == 4:
            return TrapezoidPlane(points[0], points[1], points[2], points[3], eps=eps)
        raise InvalidPlaneError(
            f"Plane must be defined by 3 or 4 3D points; got {len(points)}."
        )

    @staticmethod
    def from_coordinates(coords: Sequence[float], eps: float = EPS) -> Plane:
        """Factory method creating a Plane from 9 (3 points) or 12 (4 points) floats."""
        if len(coords) == 9:
            pts = [
                Point3D(coords[0], coords[1], coords[2]),
                Point3D(coords[3], coords[4], coords[5]),
                Point3D(coords[6], coords[7], coords[8]),
            ]
            return TrianglePlane(pts[0], pts[1], pts[2], eps=eps)
        if len(coords) == 12:
            pts = [
                Point3D(coords[0], coords[1], coords[2]),
                Point3D(coords[3], coords[4], coords[5]),
                Point3D(coords[6], coords[7], coords[8]),
                Point3D(coords[9], coords[10], coords[11]),
            ]
            return TrapezoidPlane(pts[0], pts[1], pts[2], pts[3], eps=eps)
        raise InvalidPlaneError(
            f"Expected 9 floats (3 points) or 12 floats (4 points), got {len(coords)}."
        )


class TrianglePlane(Plane):
    """3-Point Plane defined by an isosceles triangle (a, b, c).

    Validation Conditions:
        - |dist(a, b) - dist(a, c)| < EPS
        - D_ab >= 1.0 and D_bc >= 1.0 (non-degenerate dimensions)
    """

    def __init__(
        self,
        a: Point3D,
        b: Point3D,
        c: Point3D,
        eps: float = EPS,
    ) -> None:
        d_ab = a.dist(b)
        d_ac = a.dist(c)
        d_bc = b.dist(c)

        if d_ab <= eps or d_bc <= eps:
            raise InvalidPlaneError(
                f"Degenerate triangle: dist(a,b)={d_ab:.6f}, dist(b,c)={d_bc:.6f}."
            )

        if abs(d_ab - d_ac) >= eps:
            raise InvalidPlaneError(
                f"Triangle (a, b, c) is not isosceles within EPS={eps}: "
                f"|dist(a,b)={d_ab:.6f} - dist(a,c)={d_ac:.6f}| = {abs(d_ab - d_ac):.6f} >= {eps}."
            )

        # Check collinearity: triangle area must be non-zero
        cross_norm = (b - a).cross(c - a).norm()
        if cross_norm <= eps:
            raise InvalidPlaneError("Points (a, b, c) are collinear and do not define a plane.")

        super().__init__(
            a_start1=a,
            a_start2=a,
            b=b,
            c=c,
            d_ab=d_ab,
            d_bc=d_bc,
        )
        self._a = a

    @property
    def a(self) -> Point3D:
        """Apex point a."""
        return self._a

    def points(self) -> Tuple[Point3D, Point3D, Point3D]:
        return (self._a, self._b, self._c)


class TrapezoidPlane(Plane):
    """4-Point Plane defined by an isosceles trapezoid (a1, a2, b, c).

    Validation Conditions:
        - Points (a1, a2, b, c) are coplanar within EPS.
        - |dist(a1, b) - dist(a2, c)| < EPS
        - dist(a1, a2) <= dist(b, c)
    """

    def __init__(
        self,
        a1: Point3D,
        a2: Point3D,
        b: Point3D,
        c: Point3D,
        eps: float = EPS,
    ) -> None:
        d_a1b = a1.dist(b)
        d_a2c = a2.dist(c)
        d_a1a2 = a1.dist(a2)
        d_bc = b.dist(c)

        if d_a1b <= eps or d_bc <= eps:
            raise InvalidPlaneError(
                f"Degenerate trapezoid: dist(a1,b)={d_a1b:.6f}, dist(b,c)={d_bc:.6f}."
            )

        if abs(d_a1b - d_a2c) >= eps:
            raise InvalidPlaneError(
                f"Trapezoid legs are not equal within EPS={eps}: "
                f"|dist(a1,b)={d_a1b:.6f} - dist(a2,c)={d_a2c:.6f}| = {abs(d_a1b - d_a2c):.6f} >= {eps}."
            )

        if d_a1a2 > d_bc + eps:
            raise InvalidPlaneError(
                f"Trapezoid top width dist(a1,a2)={d_a1a2:.6f} exceeds base dist(b,c)={d_bc:.6f}."
            )

        # Verify coplanarity of (a1, a2, b, c)
        # Plane normal defined by (b - a1) x (c - a1)
        normal = (b - a1).cross(c - a1)
        normal_len = normal.norm()
        if normal_len <= eps:
            raise InvalidPlaneError(
                "Points (a1, b, c) are collinear; cannot form a valid 2D plane."
            )

        # Perpendicular distance from a2 to the plane spanned by (a1, b, c)
        plane_distance = abs((a2 - a1).dot(normal)) / normal_len
        if plane_distance >= eps:
            raise InvalidPlaneError(
                f"Points (a1, a2, b, c) are not coplanar within EPS={eps}: "
                f"out-of-plane distance = {plane_distance:.6f} >= {eps}."
            )

        super().__init__(
            a_start1=a1,
            a_start2=a2,
            b=b,
            c=c,
            d_ab=d_a1b,
            d_bc=d_bc,
        )
        self._a1 = a1
        self._a2 = a2

    @property
    def a1(self) -> Point3D:
        """Top-left point a1."""
        return self._a1

    @property
    def a2(self) -> Point3D:
        """Top-right point a2."""
        return self._a2

    def points(self) -> Tuple[Point3D, Point3D, Point3D, Point3D]:
        return (self._a1, self._a2, self._b, self._c)
