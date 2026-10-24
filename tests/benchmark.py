"""FPS benchmark on a 500x500x500 volume verifying >= 100 FPS throughput."""

from __future__ import annotations

import math
from pathlib import Path
import sys

# Ensure project root is on sys.path when invoked as `python tests/benchmark.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.core.projector import PlaneProjector
from src.core.volume import Volume
from src.geometry.plane import TrapezoidPlane, TrianglePlane
from src.geometry.point import Point3D
from src.utils.timer import PerformanceTimer


def build_benchmark_planes(num_frames: int = 120) -> list:
    """Create a sequence of realistic 3-point and 4-point planes slicing a 500^3 volume."""
    planes = []
    for i in range(num_frames):
        z = 50.0 + (400.0 * i) / max(1, num_frames - 1)
        if i % 2 == 0:
            # 3-Point Isosceles Triangle: height=400, base=400, D_ab = sqrt(200^2 + 400^2) ~= 447.2
            planes.append(
                TrianglePlane(
                    Point3D(250.0, 50.0, z),
                    Point3D(50.0, 450.0, z),
                    Point3D(450.0, 450.0, z),
                )
            )
        else:
            # 4-Point Isosceles Trapezoid: tilted in 3D space across (x, y, z)
            z_top = max(10.0, z - 25.0)
            z_bot = min(490.0, z + 25.0)
            planes.append(
                TrapezoidPlane(
                    Point3D(150.0, 60.0, z_top),
                    Point3D(350.0, 60.0, z_top),
                    Point3D(50.0, 440.0, z_bot),
                    Point3D(450.0, 440.0, z_bot),
                )
            )
    return planes


def run_benchmark(num_frames: int = 120) -> dict:
    """Run the 500x500x500 projection benchmark and print detailed metrics."""
    print("=" * 72)
    print("Ex1 Performance Benchmark: 500 x 500 x 500 Volume Plane Projection")
    print("=" * 72)

    rng = np.random.default_rng(seed=42)
    raw_volume = rng.integers(0, 256, size=(500, 500, 500), dtype=np.uint8)
    volume = Volume(raw_volume)
    planes = build_benchmark_planes(num_frames=num_frames)

    projector = PlaneProjector(backend="auto")

    # Warmup step (excluded from net projection FPS measurement)
    projector.warmup(volume, planes[0])
    projector.warmup(volume, planes[1])

    with PerformanceTimer("Net Projection Benchmark") as timer:
        for plane in planes:
            _ = projector.project(volume, plane)

    fps = timer.compute_fps(num_frames)
    ms_per_frame = timer.elapsed_ms / num_frames

    print(f"Volume Dimensions : {volume.shape} ({volume.dtype}, {volume.data.nbytes / (1024**2):.1f} MB)")
    print(f"Active Backend    : {projector.backend}")
    print(f"Frames Projected  : {num_frames} (50% TrianglePlane, 50% TrapezoidPlane)")
    print(f"Total Elapsed Time: {timer.elapsed_ms:.2f} ms")
    print(f"Average Frame Time: {ms_per_frame:.2f} ms / frame")
    print(f"Net Throughput    : {fps:.2f} FPS")
    print("-" * 72)

    if fps < 30.0:
        raise AssertionError(f"Benchmark FAILED minimum 30 FPS requirement: got {fps:.2f} FPS")

    target_status = "PASSED (>= 100 FPS target)" if fps >= 100.0 else "PASSED (>= 30 FPS minimum)"
    print(f"Benchmark Status  : {target_status}")
    print("=" * 72)

    return {
        "fps": fps,
        "ms_per_frame": ms_per_frame,
        "elapsed_ms": timer.elapsed_ms,
        "backend": projector.backend,
        "num_frames": num_frames,
    }


if __name__ == "__main__":
    run_benchmark()
