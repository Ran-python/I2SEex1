"""Unit and integration tests for Volume bounds safety, PlaneProjector accuracy, MRHead dataset, and MP4 output."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import numpy as np

from src.core.projector import HAS_NUMBA, PlaneProjector
from src.core.volume import Volume
from src.geometry.plane import Plane, TrapezoidPlane, TrianglePlane
from src.geometry.point import Point3D
from src.io.csv_parser import CSVPlaneParser
from src.io.nrrd_loader import DEFAULT_MRHEAD_PATH, NRRDLoader
from src.io.video_writer import HAS_CV2, VideoWriter
from src.main import run_pipeline


def reference_scalar_project(volume: Volume, plane: Plane) -> np.ndarray:
    """Exact scalar reference implementation of the specification in Section 2."""
    d_ab = plane.d_ab
    d_bc = plane.d_bc
    height = int(d_ab)
    width = int(d_bc)
    out = np.zeros((height, width), dtype=volume.dtype)

    a1 = plane.a_start1
    a2 = plane.a_start2
    v_ab = plane.v_ab
    v_ac = plane.v_ac

    for y in range(height):
        p1 = a1 + float(y) * v_ab
        p2 = a2 + float(y) * v_ac
        d12 = p1.dist(p2)
        dp = (p2 - p1) / d12 if d12 > 0.0 else Point3D(0.0, 0.0, 0.0)

        for x1 in range(int(d12) + 1):
            p = p1 + float(x1) * dp
            val = volume.sample(p.x, p.y, p.z)
            x = int((d_bc / 2.0) - (d12 / 2.0) + x1)
            if 0 <= x < width:
                out[y, x] = val

    return out


class TestProjectorAndBounds(unittest.TestCase):
    """Tests for PlaneProjector mathematical equivalence and out-of-bounds safety."""

    def setUp(self) -> None:
        sx, sy, sz = 64, 64, 64
        grid_x, grid_y, grid_z = np.meshgrid(
            np.arange(sx, dtype=np.uint16),
            np.arange(sy, dtype=np.uint16),
            np.arange(sz, dtype=np.uint16),
            indexing="ij",
        )
        data = (grid_x * 100 + grid_y * 10 + grid_z + 1).astype(np.uint16)
        self.volume = Volume(data)
        self.projector = PlaneProjector(backend="numpy")

    def test_triangle_projection_matches_scalar_reference(self) -> None:
        plane = TrianglePlane(
            Point3D(32.0, 5.0, 20.0),
            Point3D(12.0, 45.0, 20.0),
            Point3D(52.0, 45.0, 20.0),
        )
        expected = reference_scalar_project(self.volume, plane)
        actual = self.projector.project(self.volume, plane)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual, expected)

    def test_trapezoid_projection_matches_scalar_reference(self) -> None:
        plane = TrapezoidPlane(
            Point3D(22.0, 10.0, 15.0),
            Point3D(42.0, 10.0, 15.0),
            Point3D(10.0, 45.0, 35.0),
            Point3D(54.0, 45.0, 35.0),
        )
        expected = reference_scalar_project(self.volume, plane)
        actual = self.projector.project(self.volume, plane)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual, expected)

    def test_completely_out_of_bounds_returns_zeros(self) -> None:
        plane = TrianglePlane(
            Point3D(-100.0, -50.0, 200.0),
            Point3D(-130.0, -10.0, 200.0),
            Point3D(-70.0, -10.0, 200.0),
        )
        result = self.projector.project(self.volume, plane)
        self.assertEqual(result.shape, (50, 60))
        self.assertEqual(int(np.sum(result)), 0)

    def test_partially_out_of_bounds_clamps_cleanly_to_zero(self) -> None:
        plane = TrapezoidPlane(
            Point3D(-10.0, 20.0, 30.0),
            Point3D(10.0, 20.0, 30.0),
            Point3D(-30.0, 80.0, 30.0),
            Point3D(30.0, 80.0, 30.0),
        )
        expected = reference_scalar_project(self.volume, plane)
        actual = self.projector.project(self.volume, plane)
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(int(np.max(actual)), 0)
        self.assertTrue(np.any(actual == 0))

    def test_supported_bit_depths_8_16_32(self) -> None:
        for dtype in (np.uint8, np.uint16, np.uint32):
            arr = np.full((16, 16, 16), 42, dtype=dtype)
            vol = Volume(arr)
            plane = TrianglePlane(
                Point3D(8.0, 2.0, 8.0),
                Point3D(2.0, 10.0, 8.0),
                Point3D(14.0, 10.0, 8.0),
            )
            res = self.projector.project(vol, plane)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(int(np.max(res)), 42)


class TestIOPipeline(unittest.TestCase):
    """Tests for NRRDLoader, CSVPlaneParser, and VideoWriter."""

    def test_nrrd_roundtrip_and_video_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            nrrd_file = tmp_path / "test_vol.nrrd"
            csv_file = tmp_path / "test_planes.csv"
            mp4_file = tmp_path / "test_out.mp4"

            data = np.random.randint(0, 4096, size=(32, 32, 32), dtype=np.uint16)
            vol = Volume(data)
            NRRDLoader.save(nrrd_file, vol)

            loaded_vol = NRRDLoader.load(nrrd_file)
            self.assertEqual(loaded_vol.shape, (32, 32, 32))
            self.assertEqual(loaded_vol.dtype, np.uint16)
            np.testing.assert_array_equal(loaded_vol.data, vol.data)

            planes = [
                TrianglePlane(
                    Point3D(16.0, 2.0, 10.0),
                    Point3D(4.0, 26.0, 10.0),
                    Point3D(28.0, 26.0, 10.0),
                ),
                TrapezoidPlane(
                    Point3D(10.0, 4.0, 20.0),
                    Point3D(22.0, 4.0, 20.0),
                    Point3D(4.0, 28.0, 20.0),
                    Point3D(28.0, 28.0, 20.0),
                ),
            ]
            CSVPlaneParser.write(csv_file, planes)
            parsed_planes = CSVPlaneParser.parse(csv_file)
            self.assertEqual(len(parsed_planes), 2)

            projector = PlaneProjector()
            frames = [projector.project(loaded_vol, p) for p in parsed_planes]
            writer = VideoWriter(fps=30.0)
            writer.write_video(mp4_file, frames)

            info = VideoWriter.inspect_video(mp4_file)
            self.assertTrue(info["has_ftyp"])
            self.assertTrue(info["has_moov"])
            self.assertTrue(info["has_mdat"])
            self.assertGreater(info["file_size_bytes"], 64)
            if HAS_CV2:
                self.assertTrue(info["is_decodable"])
                self.assertEqual(info["frame_count"], 2)


class TestMRHeadReferenceDataset(unittest.TestCase):
    """End-to-end validation using the primary MRHead reference dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mrhead_path = NRRDLoader.resolve_path(DEFAULT_MRHEAD_PATH)
        cls.volume = NRRDLoader.load(cls.mrhead_path)

    def test_mrhead_metadata_and_value_range(self) -> None:
        # Verify MRHead dimensions, dtype conversion (int16 -> uint16), and intensity range
        self.assertEqual(self.volume.shape, (256, 256, 130))
        self.assertEqual(self.volume.dtype, np.uint16)
        self.assertEqual(self.volume.bit_depth, 16)
        self.assertEqual(int(np.min(self.volume.data)), 0)
        self.assertEqual(int(np.max(self.volume.data)), 279)

        # Verify logical alias resolution ('MRHead' and 'data/MRHead.nrrd')
        self.assertEqual(NRRDLoader.resolve_path("MRHead"), self.mrhead_path)
        self.assertEqual(NRRDLoader.resolve_path("data/MRHead.nrrd"), self.mrhead_path)

    def test_mrhead_projection_backends_match_scalar_reference(self) -> None:
        tri_plane = TrianglePlane(
            Point3D(128.0, 12.0, 65.0),
            Point3D(16.0, 236.0, 65.0),
            Point3D(240.0, 236.0, 65.0),
        )
        trap_plane = TrapezoidPlane(
            Point3D(64.0, 20.0, 50.0),
            Point3D(192.0, 20.0, 50.0),
            Point3D(16.0, 236.0, 74.0),
            Point3D(240.0, 236.0, 74.0),
        )

        np_proj = PlaneProjector(backend="numpy")
        for plane in (tri_plane, trap_plane):
            ref_slice = reference_scalar_project(self.volume, plane)
            np_slice = np_proj.project(self.volume, plane)
            np.testing.assert_array_equal(np_slice, ref_slice)
            self.assertGreater(int(np.max(np_slice)), 100)

            if HAS_NUMBA:
                nb_proj = PlaneProjector(backend="numba")
                nb_slice = nb_proj.project(self.volume, plane)
                np.testing.assert_array_equal(nb_slice, ref_slice)

    def test_mrhead_end_to_end_mp4_pipeline_and_decode(self) -> None:
        csv_path = Path("data/sample_planes.csv")
        self.assertTrue(csv_path.is_file())

        with tempfile.TemporaryDirectory() as tmp_dir:
            mp4_out = Path(tmp_dir) / "mrhead_validation.mp4"
            result = run_pipeline(
                nrrd_path=self.mrhead_path,
                csv_path=csv_path,
                mp4_path=mp4_out,
                fps=30.0,
                verbose=False,
            )

            self.assertEqual(result["volume_shape"], (256, 256, 130))
            self.assertEqual(result["num_planes"], 30)

            info = VideoWriter.inspect_video(mp4_out)
            self.assertTrue(info["has_ftyp"])
            self.assertTrue(info["has_moov"])
            self.assertTrue(info["has_mdat"])
            self.assertGreater(info["file_size_bytes"], 10_000)

            if HAS_CV2:
                self.assertIn("avc1", info["container_codec"])
                self.assertTrue(info["is_decodable"])
                self.assertEqual(info["frame_count"], 30)
                self.assertAlmostEqual(info["fps"], 30.0, places=1)
                self.assertAlmostEqual(info["duration_sec"], 1.0, places=1)
                self.assertEqual((info["height"], info["width"]), (250, 224))

                # Ensure every decoded frame is non-blank and has valid anatomical dynamic range
                for idx, frame in enumerate(info["decoded_frames"]):
                    self.assertEqual(frame.shape, (250, 224, 3))
                    self.assertEqual(frame.dtype, np.uint8)
                    self.assertGreater(
                        int(np.max(frame)),
                        150,
                        f"Decoded frame {idx} appears blank or corrupted.",
                    )
                    self.assertGreater(
                        float(np.std(frame)),
                        20.0,
                        f"Decoded frame {idx} lacks expected anatomical contrast.",
                    )


class TestMP4VideoRegression(unittest.TestCase):
    """Regression tests for MP4 normalization, NaN/Inf handling, and codec compatibility."""

    def test_normalize_handles_nan_inf_and_zero_frames(self) -> None:
        nan_frame = np.array([[np.nan, np.inf], [-np.inf, 100.0]], dtype=np.float32)
        norm = VideoWriter.normalize_to_uint8(nan_frame)
        self.assertEqual(norm.dtype, np.uint8)
        self.assertTrue(np.all(np.isfinite(norm)))
        self.assertEqual(int(norm[1, 1]), 255)
        self.assertEqual(int(norm[0, 0]), 0)

        zero_frame = np.zeros((32, 32), dtype=np.uint16)
        norm_zero = VideoWriter.normalize_to_uint8(zero_frame)
        self.assertEqual(norm_zero.dtype, np.uint8)
        self.assertEqual(int(np.max(norm_zero)), 0)

    def test_empty_frame_sequence_raises_value_error(self) -> None:
        writer = VideoWriter(fps=30.0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                writer.write_video(Path(tmp_dir) / "empty.mp4", [])

    def test_small_and_odd_dimension_frames_encode_valid_mp4(self) -> None:
        # 15x17 is both odd-sized and smaller than hardware H.264 24x24 minimum
        frames = [
            np.full((15, 17), fill_value=val, dtype=np.uint8)
            for val in (50, 120, 200, 250)
        ]
        writer = VideoWriter(fps=24.0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "odd_small.mp4"
            writer.write_video(out_path, frames)
            info = VideoWriter.inspect_video(out_path)
            self.assertTrue(info["has_moov"])
            if HAS_CV2:
                self.assertTrue(info["is_decodable"])
                self.assertEqual(info["frame_count"], 4)
                self.assertEqual(info["width"] % 2, 0)
                self.assertEqual(info["height"] % 2, 0)


if __name__ == "__main__":
    unittest.main()
