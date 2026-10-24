"""2D matrix 8-bit normalization and broadly compatible H.264 MP4 video writer."""

from __future__ import annotations

from pathlib import Path
import struct
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import zlib
import numpy as np

try:
    import cv2 as _cv2

    HAS_CV2 = True
except ImportError:
    _cv2 = None
    HAS_CV2 = False

try:
    import imageio.v2 as _imageio

    HAS_IMAGEIO = True
except ImportError:
    _imageio = None
    HAS_IMAGEIO = False

# Minimum even dimension for hardware/OS H.264 encoders (e.g. Media Foundation mfh264enc)
MIN_VIDEO_DIMENSION: int = 24


class VideoWriter:
    """Normalizes 2D projection matrices (8/16/32-bit) to 8-bit grayscale and writes .mp4 video."""

    def __init__(self, fps: float = 30.0) -> None:
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"FPS must be a finite positive number, got {fps}.")
        self.fps = float(fps)

    @staticmethod
    def normalize_to_uint8(
        frame: np.ndarray,
        global_max: Optional[float] = None,
    ) -> np.ndarray:
        """Normalize an 8-bit, 16-bit, 32-bit, or float 2D slice into an 8-bit (0-255) uint8 image.

        - Sanitizes any NaN or infinite values to 0.
        - If frame is already uint8 and global_max is None, returns a contiguous uint8 array directly.
        - For uint16, uint32, or float frames, linearly scales intensities into [0, 255] uint8.
        """
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)

        if frame.ndim != 2:
            raise ValueError(f"Expected 2D frame array, got {frame.ndim}D with shape {frame.shape}.")

        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            return np.zeros(frame.shape, dtype=np.uint8)

        # Sanitize NaN / Inf values if floating point
        if np.issubdtype(frame.dtype, np.floating):
            if not np.all(np.isfinite(frame)):
                frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)

        if frame.dtype == np.uint8 and global_max is None:
            return np.ascontiguousarray(frame)

        max_val = float(global_max) if (global_max is not None and global_max > 0.0) else float(np.max(frame))
        if not np.isfinite(max_val) or max_val <= 0.0:
            return np.zeros(frame.shape, dtype=np.uint8)

        min_val = float(np.min(frame))
        if not np.isfinite(min_val) or min_val < 0.0:
            min_val = 0.0

        if global_max is None and max_val > min_val and min_val > 0.0:
            scaled = (frame.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))
        else:
            scaled = frame.astype(np.float32) * (255.0 / max_val)

        return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def pad_to_dimensions(
        frame: np.ndarray, target_height: int, target_width: int
    ) -> np.ndarray:
        """Center-pad a 2D uint8 frame to (target_height, target_width) with zeros."""
        h, w = frame.shape
        if h == target_height and w == target_width:
            return np.ascontiguousarray(frame)

        canvas = np.zeros((target_height, target_width), dtype=np.uint8)
        copy_h = min(h, target_height)
        copy_w = min(w, target_width)
        y0 = (target_height - copy_h) // 2
        x0 = (target_width - copy_w) // 2
        canvas[y0 : y0 + copy_h, x0 : x0 + copy_w] = frame[:copy_h, :copy_w]
        return canvas

    @staticmethod
    def to_bgr(frame_gray: np.ndarray) -> np.ndarray:
        """Convert a 2D uint8 grayscale frame (H, W) into a 3-channel BGR array (H, W, 3).

        Providing 3-channel BGR frames to H.264 / MP4 encoders ensures the output stream
        is encoded in universally supported 3-plane yuv420p format rather than monochrome yuv400p.
        """
        if frame_gray.ndim == 3 and frame_gray.shape[2] == 3:
            return np.ascontiguousarray(frame_gray, dtype=np.uint8)
        if frame_gray.ndim != 2:
            raise ValueError(f"Expected 2D grayscale array, got shape {frame_gray.shape}.")
        if HAS_CV2:
            return _cv2.cvtColor(frame_gray, _cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(np.repeat(frame_gray[:, :, np.newaxis], 3, axis=2))

    def write_video(
        self,
        output_path: Union[str, Path],
        frames: Sequence[np.ndarray],
    ) -> Path:
        """Encode a sequence of 2D slices into a playable H.264 .mp4 file in sequential order."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not frames:
            raise ValueError("Cannot write video: frame sequence is empty.")

        for idx, raw_frame in enumerate(frames):
            if raw_frame is None or raw_frame.ndim != 2 or raw_frame.size == 0:
                raise ValueError(
                    f"Invalid frame at index {idx}: expected non-empty 2D numpy array, "
                    f"got {None if raw_frame is None else raw_frame.shape}."
                )

        normalized_frames: List[np.ndarray] = [
            self.normalize_to_uint8(f) for f in frames
        ]

        # H.264 yuv420p requires even width and height, and >= MIN_VIDEO_DIMENSION
        max_h = max(f.shape[0] for f in normalized_frames)
        max_w = max(f.shape[1] for f in normalized_frames)
        target_h = max(MIN_VIDEO_DIMENSION, max_h + (max_h % 2))
        target_w = max(MIN_VIDEO_DIMENSION, max_w + (max_w % 2))

        uniform_gray = [
            self.pad_to_dimensions(f, target_h, target_w) for f in normalized_frames
        ]
        uniform_bgr = [self.to_bgr(f) for f in uniform_gray]

        if HAS_CV2:
            if self._write_with_opencv(path, uniform_bgr, target_w, target_h):
                return path

        if HAS_IMAGEIO:
            try:
                with _imageio.get_writer(str(path), fps=self.fps) as writer:
                    for f_bgr in uniform_bgr:
                        writer.append_data(f_bgr)
                if self._is_valid_mp4_file(path, expected_frames=len(uniform_bgr)):
                    return path
            except Exception:
                pass

        self._write_iso_mp4_fallback(path, uniform_gray, target_w, target_h)
        return path

    def _write_with_opencv(
        self,
        path: Path,
        frames_bgr: Sequence[np.ndarray],
        width: int,
        height: int,
    ) -> bool:
        """Try encoding an H.264 (avc1) MP4 using OpenCV backends, verifying output integrity."""
        candidates: List[Tuple[int, str]] = []
        if sys.platform == "win32" and hasattr(_cv2, "CAP_MSMF"):
            # Windows Media Foundation natively encodes H.264 (avc1, Baseline yuv420p)
            # without needing external openh264 DLLs.
            candidates.append((_cv2.CAP_MSMF, "avc1"))
            candidates.append((_cv2.CAP_MSMF, "H264"))
        if sys.platform == "darwin" and hasattr(_cv2, "CAP_AVFOUNDATION"):
            candidates.append((_cv2.CAP_AVFOUNDATION, "avc1"))
        if hasattr(_cv2, "CAP_FFMPEG"):
            candidates.append((_cv2.CAP_FFMPEG, "avc1"))
            candidates.append((_cv2.CAP_FFMPEG, "mp4v"))
        candidates.append((0, "mp4v"))

        for api_pref, fourcc_str in candidates:
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

            fourcc = _cv2.VideoWriter_fourcc(*fourcc_str)
            if api_pref != 0:
                writer = _cv2.VideoWriter(
                    str(path), api_pref, fourcc, self.fps, (width, height), True
                )
            else:
                writer = _cv2.VideoWriter(
                    str(path), fourcc, self.fps, (width, height), True
                )

            if not writer.isOpened():
                writer.release()
                continue

            try:
                for frame_bgr in frames_bgr:
                    writer.write(frame_bgr)
            finally:
                writer.release()

            if self._is_valid_mp4_file(path, expected_frames=len(frames_bgr)):
                return True

        return False

    @staticmethod
    def _is_valid_mp4_file(path: Path, expected_frames: int = 1) -> bool:
        """Verify that an MP4 file exists, contains valid ISO boxes, and decodes frames."""
        if not path.is_file() or path.stat().st_size <= 64:
            return False

        raw = path.read_bytes()
        if b"ftyp" not in raw or b"moov" not in raw or b"mdat" not in raw:
            return False

        if not HAS_CV2:
            return True

        cap = _cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return False

        try:
            ret, first_frame = cap.read()
            if not ret or first_frame is None or first_frame.size == 0:
                return False
            reported_count = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            return reported_count >= expected_frames
        finally:
            cap.release()

    @staticmethod
    def inspect_video(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Inspect and decode an MP4 file, returning container metadata and decoded frames."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        file_size = path.stat().st_size
        raw = path.read_bytes()
        has_ftyp = b"ftyp" in raw
        has_moov = b"moov" in raw
        has_mdat = b"mdat" in raw

        # Detect sample entry codec tag from raw MP4 stsd atom
        container_codec = "unknown"
        for tag, label in (
            (b"avc1", "avc1 (H.264/AVC)"),
            (b"h264", "h264 (H.264/AVC)"),
            (b"mp4v", "mp4v (MPEG-4 Part 2)"),
            (b"png ", "png (ISO MP4 PNG)"),
        ):
            if tag in raw:
                container_codec = label
                break

        decoded_frames: List[np.ndarray] = []
        fps = 0.0
        width = 0
        height = 0
        fourcc_str = ""

        if HAS_CV2:
            cap = _cv2.VideoCapture(str(path))
            if cap.isOpened():
                fps = float(cap.get(_cv2.CAP_PROP_FPS))
                width = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
                fc = int(cap.get(_cv2.CAP_PROP_FOURCC))
                fourcc_str = "".join(chr((fc >> (8 * i)) & 0xFF) for i in range(4)).strip()
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    decoded_frames.append(frame)
            cap.release()

        frame_count = len(decoded_frames)
        duration_sec = (frame_count / fps) if fps > 0.0 else 0.0

        return {
            "path": str(path),
            "file_size_bytes": file_size,
            "has_ftyp": has_ftyp,
            "has_moov": has_moov,
            "has_mdat": has_mdat,
            "container_codec": container_codec,
            "decoder_fourcc": fourcc_str,
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration_sec": duration_sec,
            "is_decodable": frame_count > 0,
            "decoded_frames": decoded_frames,
        }

    def _write_iso_mp4_fallback(
        self,
        path: Path,
        frames: Sequence[np.ndarray],
        width: int,
        height: int,
    ) -> None:
        """Write a valid ISO Base Media File Format (.mp4) container using standard library only."""
        encoded_samples: List[bytes] = [
            self._encode_grayscale_png(f) for f in frames
        ]

        ftyp_payload = (
            b"isom"
            + struct.pack(">I", 512)
            + b"isomiso2mp41"
        )
        ftyp_box = self._mp4_box(b"ftyp", ftyp_payload)

        mdat_header_size = 8
        mdat_offset = len(ftyp_box) + mdat_header_size
        sample_offsets: List[int] = []
        sample_sizes: List[int] = []
        cur_offset = mdat_offset
        for sample in encoded_samples:
            sample_offsets.append(cur_offset)
            sample_sizes.append(len(sample))
            cur_offset += len(sample)

        mdat_box = self._mp4_box(b"mdat", b"".join(encoded_samples))
        moov_box = self._build_moov_box(
            width=width,
            height=height,
            fps=self.fps,
            sample_sizes=sample_sizes,
            sample_offsets=sample_offsets,
        )

        with path.open("wb") as f:
            f.write(ftyp_box)
            f.write(mdat_box)
            f.write(moov_box)

    @staticmethod
    def _mp4_box(box_type: bytes, payload: bytes) -> bytes:
        return struct.pack(">I4s", 8 + len(payload), box_type) + payload

    @staticmethod
    def _encode_grayscale_png(frame: np.ndarray) -> bytes:
        """Encode a 2D uint8 array as a valid grayscale PNG sample for MP4 'png ' sample entry."""
        height, width = frame.shape

        def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
            crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
            return struct.pack(">I4s", len(data), chunk_type) + data + struct.pack(">I", crc)

        ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
        filtered = np.empty((height, width + 1), dtype=np.uint8)
        filtered[:, 0] = 0
        filtered[:, 1:] = frame
        idat = zlib.compress(filtered.tobytes(), level=1)

        return (
            b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", ihdr)
            + png_chunk(b"IDAT", idat)
            + png_chunk(b"IEND", b"")
        )

    @classmethod
    def _build_moov_box(
        cls,
        width: int,
        height: int,
        fps: float,
        sample_sizes: Sequence[int],
        sample_offsets: Sequence[int],
    ) -> bytes:
        timescale = 1000
        frame_duration = max(1, int(round(timescale / fps)))
        num_frames = len(sample_sizes)
        total_duration = num_frames * frame_duration

        identity_matrix = struct.pack(
            ">9I",
            0x00010000, 0, 0,
            0, 0x00010000, 0,
            0, 0, 0x40000000,
        )

        # mvhd
        mvhd_payload = (
            struct.pack(">B3sIIIIIH10s", 0, b"\x00\x00\x00", 0, 0, timescale, total_duration, 0x00010000, 0x0100, b"\x00" * 10)
            + identity_matrix
            + (b"\x00" * 24)
            + struct.pack(">I", 2)
        )
        mvhd_box = cls._mp4_box(b"mvhd", mvhd_payload)

        # tkhd
        tkhd_payload = (
            struct.pack(">B3sIIIII8sHHHH", 0, b"\x00\x00\x03", 0, 0, 1, 0, total_duration, b"\x00" * 8, 0, 0, 0, 0)
            + identity_matrix
            + struct.pack(">II", width << 16, height << 16)
        )
        tkhd_box = cls._mp4_box(b"tkhd", tkhd_payload)

        # mdhd
        mdhd_payload = struct.pack(
            ">B3sIIIIHH", 0, b"\x00\x00\x00", 0, 0, timescale, total_duration, 0x55C4, 0
        )
        mdhd_box = cls._mp4_box(b"mdhd", mdhd_payload)

        # hdlr
        hdlr_payload = (
            struct.pack(">B3s4s4s12s", 0, b"\x00\x00\x00", b"\x00\x00\x00\x00", b"vide", b"\x00" * 12)
            + b"VideoHandler\x00"
        )
        hdlr_box = cls._mp4_box(b"hdlr", hdlr_payload)

        # vmhd
        vmhd_box = cls._mp4_box(b"vmhd", struct.pack(">B3sHHHH", 0, b"\x00\x00\x01", 0, 0, 0, 0))

        # dinf -> dref -> url
        url_box = cls._mp4_box(b"url ", struct.pack(">B3s", 0, b"\x00\x00\x01"))
        dref_box = cls._mp4_box(b"dref", struct.pack(">B3sI", 0, b"\x00\x00\x00", 1) + url_box)
        dinf_box = cls._mp4_box(b"dinf", dref_box)

        # stsd (png video sample entry)
        sample_entry_payload = (
            b"\x00" * 6
            + struct.pack(">H", 1)
            + b"\x00" * 16
            + struct.pack(">HHIIIH32sHH", width, height, 0x00480000, 0x00480000, 0, 1, b"\x00" * 32, 0x0018, 0xFFFF)
        )
        png_entry = cls._mp4_box(b"png ", sample_entry_payload)
        stsd_box = cls._mp4_box(b"stsd", struct.pack(">B3sI", 0, b"\x00\x00\x00", 1) + png_entry)

        # stts
        stts_box = cls._mp4_box(
            b"stts", struct.pack(">B3sIII", 0, b"\x00\x00\x00", 1, num_frames, frame_duration)
        )

        # stsc
        stsc_box = cls._mp4_box(
            b"stsc", struct.pack(">B3sIIII", 0, b"\x00\x00\x00", 1, 1, 1, 1)
        )

        # stsz
        stsz_entries = b"".join(struct.pack(">I", sz) for sz in sample_sizes)
        stsz_box = cls._mp4_box(
            b"stsz", struct.pack(">B3sII", 0, b"\x00\x00\x00", 0, num_frames) + stsz_entries
        )

        # stco
        stco_entries = b"".join(struct.pack(">I", off) for off in sample_offsets)
        stco_box = cls._mp4_box(
            b"stco", struct.pack(">B3sI", 0, b"\x00\x00\x00", num_frames) + stco_entries
        )

        stbl_box = cls._mp4_box(b"stbl", stsd_box + stts_box + stsc_box + stsz_box + stco_box)
        minf_box = cls._mp4_box(b"minf", vmhd_box + dinf_box + stbl_box)
        mdia_box = cls._mp4_box(b"mdia", mdhd_box + hdlr_box + minf_box)
        trak_box = cls._mp4_box(b"trak", tkhd_box + mdia_box)
        return cls._mp4_box(b"moov", mvhd_box + trak_box)
