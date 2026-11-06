"""Unit tests for 3D geometry primitives and plane validation rules (EPS = 0.001)."""

from __future__ import annotations

import math
import unittest

from src.geometry.point import Point3D
from src.geometry.plane import (
    EPS,
    InvalidPlaneError,
    Plane,
    TrapezoidPlane,
    TrianglePlane,
)


class TestPoint3D(unittest.TestCase):
    """Tests for Point3D vector operations and distance calculations."""

    def test_vector_arithmetic_and_distance(self) -> None:
        p1 = Point3D(1.0, 2.0, 3.0)
        p2 = Point3D(4.0, 6.0, 3.0)
        self.assertAlmostEqual(p1.dist(p2), 5.0, places=7)

        diff = p2 - p1
        self.assertEqual(diff, Point3D(3.0, 4.0, 0.0))
        self.assertAlmostEqual(diff.norm(), 5.0, places=7)

        unit = diff.normalized()
        self.assertAlmostEqual(unit.norm(), 1.0, places=7)
        self.assertAlmostEqual(unit.x, 0.6, places=7)
        self.assertAlmostEqual(unit.y, 0.8, places=7)

    def test_dot_and_cross_product(self) -> None:
        vx = Point3D(1.0, 0.0, 0.0)
        vy = Point3D(0.0, 1.0, 0.0)
        vz = vx.cross(vy)
        self.assertEqual(vz, Point3D(0.0, 0.0, 1.0))
        self.assertAlmostEqual(vx.dot(vy), 0.0, places=7)

    def test_from_sequence_and_zero_division_safety(self) -> None:
        p = Point3D.from_sequence([3.0, 4.0, 12.0])
        self.assertAlmostEqual(p.norm(), 13.0, places=7)
        self.assertEqual(Point3D(0.0, 0.0, 0.0).normalized(), Point3D(0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            Point3D.from_sequence([1.0, 2.0])
        with self.assertRaises(ZeroDivisionError):
            _ = p / 0.0


class TestTrianglePlane(unittest.TestCase):
    """Tests for 3-point Isosceles Triangle Plane validation."""

    def test_valid_isosceles_triangle(self) -> None:
        a = Point3D(50.0, 0.0, 10.0)
        b = Point3D(20.0, 40.0, 10.0)
        c = Point3D(80.0, 40.0, 10.0)
        plane = TrianglePlane(a, b, c)
        self.assertAlmostEqual(plane.d_ab, 50.0, places=5)
        self.assertAlmostEqual(plane.d_bc, 60.0, places=5)
        self.assertEqual(plane.a_start1, a)
        self.assertEqual(plane.a_start2, a)
        self.assertEqual(plane.output_dimensions, (50, 60))

    def test_triangle_within_epsilon_tolerance(self) -> None:
        a = Point3D(0.0, 0.0, 0.0)
        b = Point3D(30.0, 40.0, 0.0)  # dist = 50.0
        # Perturb c slightly so |dist(a,b) - dist(a,c)| = 0.0005 < EPS (0.001)
        c = Point3D(-30.0, 40.0005, 0.0)
        plane = TrianglePlane(a, b, c)
        self.assertLess(abs(a.dist(b) - a.dist(c)), EPS)
        self.assertIsInstance(plane, TrianglePlane)

    def test_triangle_exceeding_epsilon_rejected(self) -> None:
        a = Point3D(0.0, 0.0, 0.0)
        b = Point3D(30.0, 40.0, 0.0)  # dist = 50.0
        c = Point3D(-30.0, 40.01, 0.0)  # |dist(a,b) - dist(a,c)| > 0.001
        with self.assertRaises(InvalidPlaneError):
            TrianglePlane(a, b, c)

    def test_collinear_triangle_rejected(self) -> None:
        a = Point3D(50.0, 50.0, 50.0)
        b = Point3D(50.0, 50.0, 50.0)
        c = Point3D(50.0, 50.0, 50.0)
        with self.assertRaises(InvalidPlaneError):
            TrianglePlane(a, b, c)


class TestTrapezoidPlane(unittest.TestCase):
    """Tests for 4-point Isosceles Trapezoid Plane validation."""

    def test_valid_isosceles_trapezoid(self) -> None:
        a1 = Point3D(30.0, 10.0, 25.0)
        a2 = Point3D(70.0, 10.0, 25.0)
        b = Point3D(10.0, 50.0, 25.0)
        c = Point3D(90.0, 50.0, 25.0)
        plane = TrapezoidPlane(a1, a2, b, c)
        expected_leg = math.sqrt((30.0 - 10.0) ** 2 + (10.0 - 50.0) ** 2)
        self.assertAlmostEqual(plane.d_ab, expected_leg, places=5)
        self.assertAlmostEqual(plane.d_bc, 80.0, places=5)
        self.assertEqual(plane.a_start1, a1)
        self.assertEqual(plane.a_start2, a2)

    def test_non_coplanar_trapezoid_rejected(self) -> None:
        a1 = Point3D(30.0, 10.0, 25.0)
        a2 = Point3D(70.0, 10.0, 26.0)  # z shifted by 1.0 -> non-coplanar
        b = Point3D(10.0, 50.0, 25.0)
        c = Point3D(90.0, 50.0, 25.0)
        with self.assertRaises(InvalidPlaneError):
            TrapezoidPlane(a1, a2, b, c)

    def test_unequal_legs_trapezoid_rejected(self) -> None:
        a1 = Point3D(30.0, 10.0, 25.0)
        a2 = Point3D(70.0, 10.0, 25.0)
        b = Point3D(10.0, 50.0, 25.0)
        c = Point3D(95.0, 50.0, 25.0)  # dist(a2, c) != dist(a1, b)
        with self.assertRaises(InvalidPlaneError):
            TrapezoidPlane(a1, a2, b, c)

    def test_top_wider_than_base_rejected(self) -> None:
        # dist(a1, a2) = 80 > dist(b, c) = 40
        a1 = Point3D(10.0, 10.0, 25.0)
        a2 = Point3D(90.0, 10.0, 25.0)
        b = Point3D(30.0, 50.0, 25.0)
        c = Point3D(70.0, 50.0, 25.0)
        with self.assertRaises(InvalidPlaneError):
            TrapezoidPlane(a1, a2, b, c)

    def test_factory_from_coordinates(self) -> None:
        tri = Plane.from_coordinates([50, 0, 10, 20, 40, 10, 80, 40, 10])
        self.assertIsInstance(tri, TrianglePlane)
        trap = Plane.from_coordinates([30, 10, 25, 70, 10, 25, 10, 50, 25, 90, 50, 25])
        self.assertIsInstance(trap, TrapezoidPlane)

    def test_trapezoid_within_epsilon_and_invalid_coord_counts(self) -> None:
        a1 = Point3D(30.0, 10.0, 25.0)
        a2 = Point3D(70.0, 10.0, 25.0)
        b = Point3D(10.0, 50.0, 25.0)
        c = Point3D(90.0, 50.0004, 25.0)  # Leg diff < EPS (0.001)
        trap = TrapezoidPlane(a1, a2, b, c)
        self.assertIsInstance(trap, TrapezoidPlane)

        with self.assertRaises(InvalidPlaneError):
            Plane.from_coordinates([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        with self.assertRaises(InvalidPlaneError):
            Plane.from_points([a1, a2])


if __name__ == "__main__":
    unittest.main()
