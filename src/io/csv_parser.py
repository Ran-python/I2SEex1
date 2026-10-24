"""CSV parser for 3-point (Triangle) and 4-point (Trapezoid) plane definitions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence, Union

from src.geometry.plane import EPS, InvalidPlaneError, Plane


class CSVPlaneParser:
    """Parses rows of 3D coordinates from a CSV file into validated Plane objects."""

    @staticmethod
    def parse(file_path: Union[str, Path], eps: float = EPS) -> List[Plane]:
        """Parse a CSV file where each row defines 3 points (9 floats) or 4 points (12 floats).

        Blank lines, comment lines starting with '#', and optional text header rows
        are automatically skipped.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

        planes: List[Plane] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for line_num, row in enumerate(reader, start=1):
                cleaned = [cell.strip() for cell in row if cell.strip() != ""]
                if not cleaned or cleaned[0].startswith("#"):
                    continue

                # Skip optional header row if first token is non-numeric
                if line_num == 1 and not CSVPlaneParser._is_float(cleaned[0]):
                    continue

                try:
                    coords = [float(val) for val in cleaned]
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric coordinate in {path.name} at line {line_num}: {row}"
                    ) from exc

                if len(coords) not in (9, 12):
                    raise InvalidPlaneError(
                        f"Invalid coordinate count in {path.name} at line {line_num}: "
                        f"expected 9 (3 points) or 12 (4 points) values, got {len(coords)}."
                    )

                try:
                    plane = Plane.from_coordinates(coords, eps=eps)
                except InvalidPlaneError as exc:
                    raise InvalidPlaneError(
                        f"Invalid plane geometry in {path.name} at line {line_num}: {exc}"
                    ) from exc

                planes.append(plane)

        return planes

    @staticmethod
    def write(file_path: Union[str, Path], planes: Sequence[Plane]) -> None:
        """Write a sequence of Plane objects to a CSV file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for plane in planes:
                row: List[float] = []
                for pt in plane.points():
                    row.extend([pt.x, pt.y, pt.z])
                writer.writerow(row)

    @staticmethod
    def _is_float(token: str) -> bool:
        try:
            float(token)
            return True
        except ValueError:
            return False
