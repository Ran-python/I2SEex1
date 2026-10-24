"""Main orchestrator coordinating IO, Geometry, and 3D Volume Plane Projection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Sequence
import numpy as np

from src.core.projector import PlaneProjector
from src.core.volume import Volume
from src.geometry.plane import Plane
from src.io.csv_parser import CSVPlaneParser
from src.io.nrrd_loader import NRRDLoader
from src.io.video_writer import VideoWriter
from src.utils.timer import PerformanceTimer


def run_pipeline(
    nrrd_path: str | Path,
    csv_path: str | Path,
    mp4_path: str | Path,
    fps: float = 30.0,
    verbose: bool = True,
) -> dict:
    """Execute the end-to-end NRRD -> Plane Projection -> MP4 pipeline."""
    with PerformanceTimer("Load NRRD Volume") as load_timer:
        volume: Volume = NRRDLoader.load(nrrd_path)

    with PerformanceTimer("Parse CSV Planes") as csv_timer:
        planes: List[Plane] = CSVPlaneParser.parse(csv_path)

    if not planes:
        raise ValueError(f"No valid plane definitions found in CSV file: {csv_path}")

    projector = PlaneProjector(backend="auto")
    # Warmup JIT / cache prior to timing net projection throughput
    projector.warmup(volume, planes[0])

    frames: List[np.ndarray] = []
    with PerformanceTimer("Project Planes") as proj_timer:
        for plane in planes:
            slice_2d = projector.project(volume, plane)
            frames.append(slice_2d)

    with PerformanceTimer("Encode MP4 Video") as video_timer:
        writer = VideoWriter(fps=fps)
        output_file = writer.write_video(mp4_path, frames)

    net_fps = proj_timer.compute_fps(len(planes))
    avg_ms = proj_timer.elapsed_ms / len(planes)

    if verbose:
        print(
            f"[Ex1 Pipeline] Volume: {volume.shape} ({volume.dtype}) | "
            f"Planes: {len(planes)} | Backend: {projector.backend}"
        )
        print(
            f"[Ex1 Timing] Load: {load_timer.elapsed_ms:.2f} ms | "
            f"CSV: {csv_timer.elapsed_ms:.2f} ms | "
            f"Projection: {proj_timer.elapsed_ms:.2f} ms ({avg_ms:.2f} ms/frame, {net_fps:.1f} FPS) | "
            f"Video: {video_timer.elapsed_ms:.2f} ms"
        )
        print(f"[Ex1 Output] Wrote {len(frames)} frames to {output_file}")

    return {
        "volume_shape": volume.shape,
        "volume_dtype": str(volume.dtype),
        "num_planes": len(planes),
        "backend": projector.backend,
        "projection_ms": proj_timer.elapsed_ms,
        "avg_frame_ms": avg_ms,
        "net_fps": net_fps,
        "output_path": str(output_file),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for Ex0.py / main.py."""
    parser = argparse.ArgumentParser(
        prog="Ex0.py",
        description="Project 2D planes (defined in CSV) from a 3D NRRD volume into an MP4 video.",
    )
    parser.add_argument("nrrd_file", type=str, help="Path to input 3D volumetric .nrrd file.")
    parser.add_argument("csv_file", type=str, help="Path to input .csv file containing 3-point or 4-point planes.")
    parser.add_argument("mp4_file", type=str, help="Path to output .mp4 video file.")
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video frame rate (default: 30.0).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console summary output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    try:
        run_pipeline(
            nrrd_path=args.nrrd_file,
            csv_path=args.csv_file,
            mp4_path=args.mp4_file,
            fps=args.fps,
            verbose=not args.quiet,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
