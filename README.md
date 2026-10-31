# Ex1: Implementing a Plane Projection of a 3D Volume

A high-performance, object-oriented Python system for slicing and projecting arbitrary 2D planes (isosceles triangles and isosceles trapezoids) from 3D volumetric raster datasets (`.nrrd`) into 2D image sequences encoded as `.mp4` videos.

---

## 1. Problem Statement & Project Scope

Given a 3D raster matrix $M$ of dimensions $s_x \times s_y \times s_z$ (where $1 \le s_x, s_y, s_z < 2^{10}$ and entries are 8-bit, 16-bit, or 32-bit unsigned integers) and a list of planar regions defined in a `.csv` file, the objective is to project each 3D plane $p$ onto a 2D image matrix $E$ and export the resulting sequence of 2D slices as frames in an `.mp4` video file.

### Plane Representations ($\text{EPS} = 0.001$)
1. **3-Point Plane (Isosceles Triangle $a, b, c$)**:
   - Condition: $|dist(a,b) - dist(a,c)| < \text{EPS}$ and non-collinear points.
   - $D_{ab} = dist(a,b)$, $D_{bc} = dist(b,c)$.
   - Starting top points: $a_{start1} = a$, $a_{start2} = a$.
2. **4-Point Plane (Isosceles Trapezoid $a_1, a_2, b, c$)**:
   - Condition: Coplanar points where $|dist(a_1,b) - dist(a_2,c)| < \text{EPS}$ and $dist(a_1,a_2) \le dist(b,c)$.
   - $D_{ab} = dist(a_1,b)$, $D_{bc} = dist(b,c)$.
   - Starting top points: $a_{start1} = a_1$, $a_{start2} = a_2$.

### Projection Algorithm
- Output 2D matrix $E$ has dimensions $\text{int}(D_{ab}) \times \text{int}(D_{bc})$ (height $\text{int}(D_{ab})$, width $\text{int}(D_{bc})$).
- Normalized leg vectors:
  $$v_{ab} = \frac{b - a_{start1}}{D_{ab}}, \quad v_{ac} = \frac{c - a_{start2}}{D_{ab}}$$
- For each vertical step $y \in [0, D_{ab})$:
  - $p_1(y) = a_{start1} + y \cdot v_{ab}$
  - $p_2(y) = a_{start2} + y \cdot v_{ac}$
  - $d_{12} = dist(p_1(y), p_2(y))$
  - $dp = \frac{p_2(y) - p_1(y)}{d_{12}}$ (safely zeroed when $d_{12} = 0$ at a triangle apex)
  - For each horizontal step $x_1 \in [0, d_{12}]$:
    - 3D sample point: $p = p_1(y) + x_1 \cdot dp$
    - Voxel value: $v = M[p]$ if $p \in [0, s_x) \times [0, s_y) \times [0, s_z)$, else $v = 0$.
    - Target 2D horizontal column index: $x = \text{int}\left(\frac{D_{bc}}{2} - \frac{d_{12}}{2} + x_1\right)$
    - Set $E[y, x] = v$ (background pixels outside the triangle/trapezoid default to `0`).

---

## 2. Project Directory Structure

```text
project_root/
├── data/                    # Reference .nrrd volume and .csv plane lists
│   ├── MRHead_1.nrrd        # Primary 256x256x130 3D Slicer MRI reference dataset
│   ├── sample_planes.csv    # 30 valid 3-point (triangle) and 4-point (trapezoid) planes
│   └── sample_planes_2.csv  # Extended plane set testing coronal, axial, and 3D oblique slices
├── docs/                    # Documentation & Software Requirements Specification
│   └── SRS.md             # Formal SRS, class diagram, and verification strategy
├── src/                   # Core source code and components
│   ├── __init__.py
│   ├── main.py            # Main orchestrator coordinating IO, Geometry, and Projection
│   ├── core/              # Volumetric data wrapper & high-speed projection engine
│   │   ├── __init__.py
│   │   ├── volume.py      # 3D matrix encapsulation (8/16/32-bit unsigned int) & bounds checking
│   │   └── projector.py   # Vectorized NumPy & Numba JIT parallel plane projection engine
│   ├── geometry/          # 3D geometric entities and validations
│   │   ├── __init__.py
│   │   ├── point.py       # Immutable 3D coordinates and vector math
│   │   └── plane.py       # Polymorphic 3-point (TrianglePlane) & 4-point (TrapezoidPlane)
│   ├── io/                # Input/Output handlers
│   │   ├── __init__.py
│   │   ├── nrrd_loader.py # Reads/writes .nrrd files into Volume objects
│   │   ├── csv_parser.py  # Parses 3-point or 4-point rows from .csv
│   │   └── video_writer.py# Normalizes 2D matrices to 8-bit grayscale and encodes .mp4
│   └── utils/             # Helpers and performance timers
│       ├── __init__.py
│       └── timer.py
├── tests/                 # Unit tests and FPS benchmarking
│   ├── test_geometry.py   # Tests epsilon tolerances, triangle & trapezoid validity
│   ├── test_projector.py  # Synthetic 3D volume tests & out-of-bounds safety (returns 0)
│   └── benchmark.py       # Measures net FPS on a 500x500x500 volume
├── Ex0.py                 # Root CLI entry point forwarding arguments to src/main.py
├── .gitignore             # Excludes venv/, .venv/, __pycache__/, *.mp4, etc.
├── README.md              # Problem statement, architecture, instructions, benchmark results
└── requirements.txt       # Project dependencies
```

---

## 3. Installation & Setup

```bash
# Optional: create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Zero-Dependency Resilience**: While `pynrrd`, `numba`, and `opencv-python` are supported out of the box, the codebase also includes native fallbacks for `.nrrd` parsing (`raw`, `gzip`, `ascii`), 2D vectorized NumPy projection, and ISO `.mp4` container encoding using only `numpy` and the Python standard library.

---

## 4. Usage (CLI Interface)

Run the projection pipeline from the project root directory using `Ex0.py`:

```bash
python Ex0.py <file.nrrd> <file.csv> <file.mp4>
```

### Example with Primary Reference Dataset (`MRHead`)
```bash
python Ex0.py data/MRHead_1.nrrd data/sample_planes_2.csv data/demo_output.mp4
```

Sample output:
```text
[Ex1 Pipeline] Volume: (256, 256, 130) (uint16) | Planes: 100 | Backend: numpy
[Ex1 Timing] Load: 266.92 ms | CSV: 2.17 ms | Projection: 108.78 ms (1.09 ms/frame, 919.2 FPS) | Video: 634.02 ms
[Ex1 Output] Wrote 100 frames to data\demo_output.mp4
Runtime: 1.0119 seconds
```

---

## 5. Running Unit Tests & Benchmarks

### Run Unit Test Suite
```bash
pytest -v
# or using standard library unittest:
python -m unittest discover -s tests -p "test_*.py" -v
```

### Run $500 \times 500 \times 500$ Performance Benchmark
```bash
python tests/benchmark.py
```

### Test Suite Logic & Verification Architecture
The automated test suite (23 unit and integration tests) validates every layer of the system:

1. **Geometric Validation Logic ([`tests/test_geometry.py`](file:///c:/projects/OOP_projects/Ex1/tests/test_geometry.py))**:
   - **`TestPoint3D`**: Verifies 3D vector subtraction, Euclidean distance (`dist`), vector normalization, dot products, and cross products.
   - **`TestTrianglePlane`**: Verifies that 3-point planes $(a, b, c)$ pass when $|dist(a,b) - dist(a,c)| < \text{EPS}$ ($\text{EPS} = 0.001$), and confirms `InvalidPlaneError` is raised when leg lengths differ by $\ge 0.001$ or when points are collinear ($\|(b-a) \times (c-a)\| \le \text{EPS}$).
   - **`TestTrapezoidPlane`**: Verifies 4-point planes $(a_1, a_2, b, c)$ for 3D coplanarity (perpendicular distance of $a_2$ from the plane of $(a_1, b, c)$ is $< \text{EPS}$), equal leg lengths ($|dist(a_1,b) - dist(a_2,c)| < \text{EPS}$), and width ordering ($dist(a_1,a_2) \le dist(b,c)$).

2. **Projection Mathematical Equivalence & Boundary Logic ([`tests/test_projector.py`](file:///c:/projects/OOP_projects/Ex1/tests/test_projector.py))**:
   - **Scalar Reference Oracle (`reference_scalar_project`)**: Implements the exact point-by-point mathematical specification from Section 1 using explicit scalar loops as a ground-truth oracle.
   - **`TestProjectorAndBounds`**: Compares the vectorized `PlaneProjector` output against `reference_scalar_project` using `np.testing.assert_array_equal` (100% pixel-for-pixel equality) on both `TrianglePlane` and `TrapezoidPlane`. Tests planes completely outside the 3D volume (verifying an all-zero matrix is returned without crashing) and planes partially straddling volume borders (verifying in-bounds voxels are sampled and out-of-bounds coordinates clamp cleanly to `0`).
   - **`TestMRHeadReferenceDataset`**: Loads `data/MRHead_1.nrrd`, verifies its $(256, 256, 130)$ dimensions, lossless `int16 -> uint16` conversion, and intensity range $[0, 279]$, and runs the full end-to-end pipeline to `.mp4`.
   - **`TestMP4VideoRegression`**: Verifies `NaN`/`Inf` sanitization during 8-bit normalization, odd/small frame dimension padding, ISO MP4 container atoms (`ftyp`, `moov`, `mdat`), H.264 (`avc1`) stream headers, and frame-by-frame decoding via `cv2.VideoCapture` to ensure frames are non-blank and preserve anatomical contrast.

---

## 6. Architecture, Timing Methodology & Benchmark Summary

### 6.1 How the Vectorized Projection Engine Works (`PlaneProjector`)
Instead of iterating pixel-by-pixel in Python loops, [`_project_numpy_vectorized`](file:///c:/projects/OOP_projects/Ex1/src/core/projector.py#L96-L192) computes each 2D slice in three vectorized C-level NumPy steps:
1. **1D Scanline Endpoints**: For all vertical steps $y \in [0, \text{int}(D_{ab}))$ simultaneously, it computes the left endpoint $p_1(y) = a_{start1} + y \cdot v_{ab}$, right endpoint $p_2(y) = a_{start2} + y \cdot v_{ac}$, scanline width $d_{12}(y) = \|p_2(y) - p_1(y)\|_2$, and unit step vector $dp(y) = \frac{p_2(y) - p_1(y)}{d_{12}(y)}$.
2. **2D Coordinate Grid Broadcast**: Using the centering offset $x_{\text{start}}(y) = \lfloor \frac{D_{bc}}{2} - \frac{d_{12}(y)}{2} \rfloor$, it broadcasts the horizontal step $x_1 = x - x_{\text{start}}(y)$ across the 2D image grid to compute the 3D voxel indices $(p_x, p_y, p_z) = \lfloor p_1(y) + x_1 \cdot dp(y) \rfloor$ in hardware vectorized arithmetic.
3. **Boundary Masking & 1D Flat Gather**: A boolean mask identifies pixels that lie both inside the triangle/trapezoid ($0 \le x_1 \le \lfloor d_{12}(y) \rfloor$) and strictly inside the 3D volume bounds ($0 \le p_x < s_x, 0 \le p_y < s_y, 0 \le p_z < s_z$). Valid coordinates are converted to 1D contiguous memory offsets (`flat_idx = px * (sy * sz) + py * sz + pz`) and gathered in a single operation (`volume.flat_data[flat_idx]`), while all out-of-bounds positions remain `0`.

### 6.2 How `Runtime` vs. `Projection FPS` Are Calculated
The CLI reports both the **total program wall-clock `Runtime`** and the **net algorithmic `Projection FPS`**, which measure two distinct quantities:

1. **Total `Runtime` (End-to-End Pipeline Wall-Clock Time)**:
   Measures the complete execution time of `run_pipeline` from start to finish, including disk I/O, decompression, projection, and H.264 video encoding:
   $$\text{Runtime} = T_{\text{Load}} + T_{\text{CSV}} + T_{\text{Projection}} + T_{\text{Video}}$$
   In the 100-plane `MRHead` run above:
   $$266.92\text{ ms (Load)} + 2.17\text{ ms (CSV)} + 108.78\text{ ms (Projection)} + 634.02\text{ ms (Video)} = 1011.89\text{ ms} \approx \mathbf{1.0119\text{ seconds}}$$

2. **Net `Projection FPS` (Algorithmic Projection Throughput)**:
   Per the assignment specification (*"Given a matrix of size $500^3$ bytes, a runtime of at least 100 fps is expected"*), `FPS` measures strictly the speed of the **3D-to-2D plane projection algorithm** (`proj_timer`), independent of disk reading or MP4 file compression:
   $$\text{Average Frame Time} = \frac{T_{\text{Projection}}}{N_{\text{planes}}}, \qquad \text{Projection FPS} = \frac{N_{\text{planes}}}{T_{\text{Projection (seconds)}}}$$
   In the 100-plane `MRHead` run:
   $$\text{Average Frame Time} = \frac{108.78\text{ ms}}{100} = \mathbf{1.09\text{ ms/frame}}, \qquad \text{Projection FPS} = \frac{100}{0.10878\text{ s}} = \mathbf{919.2\text{ FPS}}$$

### 6.3 Benchmark on $500 \times 500 \times 500$ Volume (`tests/benchmark.py`)
The benchmark evaluates 120 planar slices (60 `TrianglePlane` slices and 60 oblique 3D `TrapezoidPlane` slices) across a $500 \times 500 \times 500$ `uint8` volume ($125,000,000$ voxels, $119.2\text{ MB}$):

| Metric | Measured Result | Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Volume Size** | $500 \times 500 \times 500$ (`uint8`) | $500^3$ bytes | Verified |
| **Frames Projected** | $120$ frames ($50\%$ Triangle, $50\%$ Trapezoid) | Sequential plane list | Verified |
| **Total Projection Time** | $629.93\text{ ms}$ | $\le 4000\text{ ms}$ (for 30 FPS) | **PASSED** |
| **Average Time per Frame** | **$5.25\text{ ms / frame}$** | $< 10.0\text{ ms / frame}$ | **PASSED** |
| **Net Projection FPS** | **$190.50\text{ FPS}$** | Minimum $\ge 30\text{ FPS}$, Target $\ge 100\text{ FPS}$ | **PASSED (190+ FPS)** |

### 6.4 Demo Experiment (`data/MRHead_1.nrrd` & `data/sample_planes_2.csv`)
- **Dataset**: The 3D Slicer `MRHead` dataset (`data/MRHead_1.nrrd`), a $256 \times 256 \times 130$ T1 MRI scan stored as gzip-compressed 16-bit integer voxels (`min=0, max=279`, converted losslessly to `uint16`).
- **Trajectory**: `data/sample_planes.csv` and `data/sample_planes_2.csv` define sequential 3-point (`TrianglePlane`) and 4-point (`TrapezoidPlane`) slices sweeping across sagittal, coronal, axial, and 3D oblique orientations.
- **Experimental Result**: Running `python Ex0.py data/MRHead_1.nrrd data/sample_planes_2.csv data/demo_output.mp4` loads the volume in $266.92\text{ ms}$, parses 100 planes in $2.17\text{ ms}$, projects all 100 frames in **$108.78\text{ ms}$ ($1.09\text{ ms/frame}$, $919.2\text{ FPS}$)**, and encodes `data/demo_output.mp4` in $634.02\text{ ms}$ for a total end-to-end runtime of **$1.0119\text{ seconds}$**.
