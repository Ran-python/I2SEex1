"""NRRD volumetric file reader and writer."""

from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union
import numpy as np

from src.core.volume import Volume

try:
    import nrrd as _pynrrd

    HAS_PYNRRD = True
except ImportError:
    _pynrrd = None
    HAS_PYNRRD = False


DEFAULT_MRHEAD_PATH = Path(
    os.environ.get("MRHEAD_DATASET_PATH", "data/MRHead_1.nrrd")
)

_NRRD_TYPE_MAP: Dict[str, str] = {
    "uchar": "u1",
    "unsigned char": "u1",
    "uint8": "u1",
    "uint8_t": "u1",
    "int8": "i1",
    "int8_t": "i1",
    "signed char": "i1",
    "char": "i1",
    "ushort": "u2",
    "unsigned short": "u2",
    "unsigned short int": "u2",
    "uint16": "u2",
    "uint16_t": "u2",
    "short": "i2",
    "short int": "i2",
    "signed short": "i2",
    "signed short int": "i2",
    "int16": "i2",
    "int16_t": "i2",
    "uint": "u4",
    "unsigned int": "u4",
    "uint32": "u4",
    "uint32_t": "u4",
    "int": "i4",
    "signed int": "i4",
    "int32": "i4",
    "int32_t": "i4",
    "float": "f4",
    "double": "f8",
}

_DTYPE_TO_NRRD_TYPE: Dict[np.dtype, str] = {
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
}


class NRRDLoader:
    """Loads and saves 3D volumetric datasets in NRRD (.nrrd) format."""

    @staticmethod
    def resolve_path(file_path: Union[str, Path]) -> Path:
        """Resolve a dataset path or logical dataset alias (such as 'MRHead')."""
        path = Path(file_path)
        if path.is_file():
            return path

        # Support logical alias 'MRHead' or 'data/MRHead(.nrrd)' resolving to MRHead_1.nrrd
        stem_lower = path.stem.lower()
        search_dirs = [path.parent, Path("data")]
        if stem_lower in ("mrhead", "mrhead_1"):
            for directory in search_dirs:
                for candidate_name in ("MRHead_1.nrrd", "MRHead.nrrd"):
                    candidate = directory / candidate_name
                    if candidate.is_file():
                        return candidate

        if path.suffix == "":
            with_ext = path.with_suffix(".nrrd")
            if with_ext.is_file():
                return with_ext

        raise FileNotFoundError(f"NRRD file not found: {path}")

    @staticmethod
    def load(file_path: Union[str, Path] = DEFAULT_MRHEAD_PATH) -> Volume:
        """Read a .nrrd file from disk and encapsulate it in a Volume object."""
        path = NRRDLoader.resolve_path(file_path)

        if HAS_PYNRRD:
            raw_data, header = _pynrrd.read(str(path))
            header_dict: Dict[str, Any] = dict(header)
        else:
            raw_data, header_dict = NRRDLoader._read_nrrd_native(path)

        normalized_data = NRRDLoader._ensure_unsigned_dtype(raw_data)
        return Volume(data=normalized_data, header=header_dict)

    @staticmethod
    def save(file_path: Union[str, Path], volume: Volume, compressed: bool = False) -> None:
        """Write a Volume object to disk as a standard .nrrd file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_PYNRRD:
            _pynrrd.write(str(path), volume.data)
            return

        NRRDLoader._write_nrrd_native(path, volume.data, compressed=compressed)

    @staticmethod
    def _ensure_unsigned_dtype(arr: np.ndarray) -> np.ndarray:
        """Ensure array dtype is uint8, uint16, or uint32 as required by Volume."""
        if arr.dtype in (np.uint8, np.uint16, np.uint32):
            return np.ascontiguousarray(arr)

        if np.issubdtype(arr.dtype, np.signedinteger):
            # Shift negative values (if any) in signed integer CT/MRI datasets to >= 0
            min_val = int(np.min(arr)) if arr.size > 0 else 0
            shifted = arr.astype(np.int64) - min_val if min_val < 0 else arr.astype(np.int64)
            max_val = int(np.max(shifted)) if shifted.size > 0 else 0
            if max_val <= 255 and arr.itemsize == 1:
                return np.ascontiguousarray(shifted.astype(np.uint8))
            if max_val <= 65535 and arr.itemsize <= 2:
                return np.ascontiguousarray(shifted.astype(np.uint16))
            return np.ascontiguousarray(shifted.astype(np.uint32))

        if np.issubdtype(arr.dtype, np.floating):
            clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            clipped = np.clip(clean, 0.0, float(np.iinfo(np.uint32).max))
            return np.ascontiguousarray(clipped.astype(np.uint32))

        raise TypeError(f"Unsupported NRRD array data type: {arr.dtype}")

    @staticmethod
    def _read_nrrd_native(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Native NRRD parser supporting raw, gzip, and ascii encodings."""
        raw_bytes = path.read_bytes()
        if not raw_bytes.startswith(b"NRRD"):
            raise ValueError(f"Invalid NRRD file (missing NRRD magic header): {path}")

        sep_idx = -1
        sep_len = 0
        for token in (b"\r\n\r\n", b"\n\n", b"\r\r"):
            idx = raw_bytes.find(token)
            if idx != -1 and (sep_idx == -1 or idx < sep_idx):
                sep_idx = idx
                sep_len = len(token)

        if sep_idx == -1:
            raise ValueError(f"Invalid NRRD file (missing header-data separator): {path}")

        header_text = raw_bytes[:sep_idx].decode("ascii", errors="replace")
        payload = raw_bytes[sep_idx + sep_len :]

        header: Dict[str, Any] = {}
        for line in header_text.splitlines()[1:]:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            if ":=" in line_str:
                k, v = line_str.split(":=", 1)
                header[k.strip().lower()] = v.strip()
            elif ":" in line_str:
                k, v = line_str.split(":", 1)
                header[k.strip().lower()] = v.strip()

        dimension = int(header.get("dimension", "3"))
        if dimension != 3:
            raise ValueError(f"Expected 3D NRRD volume, got dimension={dimension}.")

        sizes_str = header.get("sizes", "")
        sizes = tuple(int(s) for s in sizes_str.split())
        if len(sizes) != 3:
            raise ValueError(f"Invalid NRRD 'sizes' field: {sizes_str!r}")

        nrrd_type = header.get("type", "").lower()
        if nrrd_type not in _NRRD_TYPE_MAP:
            raise ValueError(f"Unsupported NRRD 'type' field: {nrrd_type!r}")
        base_type = _NRRD_TYPE_MAP[nrrd_type]

        endian = header.get("endian", "little").lower()
        endian_prefix = "<" if endian == "little" else (">" if endian == "big" else "|")
        if base_type.endswith("1"):
            endian_prefix = "|"
        dtype = np.dtype(f"{endian_prefix}{base_type}")

        encoding = header.get("encoding", "raw").lower()
        if encoding in ("gzip", "gz"):
            payload = gzip.decompress(payload)
            arr_1d = np.frombuffer(payload, dtype=dtype)
        elif encoding == "raw":
            arr_1d = np.frombuffer(payload, dtype=dtype)
        elif encoding in ("ascii", "text", "txt"):
            text_payload = payload.decode("ascii")
            arr_1d = np.fromstring(text_payload, dtype=dtype, sep=" ")
        else:
            raise ValueError(f"Unsupported NRRD encoding: {encoding!r}")

        expected_count = sizes[0] * sizes[1] * sizes[2]
        if arr_1d.size != expected_count:
            raise ValueError(
                f"NRRD payload size mismatch: expected {expected_count} voxels, got {arr_1d.size}."
            )

        arr_3d = np.ascontiguousarray(arr_1d.reshape(sizes, order="F"))
        return arr_3d, header

    @staticmethod
    def _write_nrrd_native(path: Path, data: np.ndarray, compressed: bool = False) -> None:
        """Native NRRD writer producing standard NRRD0004 files."""
        nrrd_type = _DTYPE_TO_NRRD_TYPE.get(data.dtype)
        if nrrd_type is None:
            raise TypeError(f"Cannot write NRRD for dtype {data.dtype}")

        sx, sy, sz = data.shape
        encoding = "gzip" if compressed else "raw"
        header_lines = [
            "NRRD0004",
            "# Complete NRRD file format specification at:",
            "# http://teem.sourceforge.net/nrrd/format.html",
            f"type: {nrrd_type}",
            "dimension: 3",
            f"sizes: {sx} {sy} {sz}",
            "endian: little",
            f"encoding: {encoding}",
            "",
            "",
        ]
        header_bytes = "\n".join(header_lines).encode("ascii")
        raw_payload = np.asfortranarray(data.astype(data.dtype.newbyteorder("<"))).tobytes(order="F")
        if compressed:
            raw_payload = gzip.compress(raw_payload)

        with path.open("wb") as f:
            f.write(header_bytes)
            f.write(raw_payload)
