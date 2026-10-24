"""High-resolution performance timer for FPS and latency benchmarking."""

from __future__ import annotations

import time
from typing import Optional


class PerformanceTimer:
    """Context-manager timer using time.perf_counter() for sub-microsecond accuracy."""

    def __init__(self, name: str = "Timer") -> None:
        self.name = name
        self._start_time: Optional[float] = None
        self.elapsed_seconds: float = 0.0

    def start(self) -> PerformanceTimer:
        """Start the timer."""
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds."""
        if self._start_time is None:
            raise RuntimeError("PerformanceTimer.stop() called before start().")
        self.elapsed_seconds = time.perf_counter() - self._start_time
        self._start_time = None
        return self.elapsed_seconds

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        return self.elapsed_seconds * 1000.0

    def compute_fps(self, num_frames: int) -> float:
        """Compute frames per second for a given frame count."""
        if self.elapsed_seconds <= 0.0:
            return 0.0
        return float(num_frames) / self.elapsed_seconds

    def __enter__(self) -> PerformanceTimer:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
