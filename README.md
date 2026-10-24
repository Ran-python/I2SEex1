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
├── data/                  # Reference .nrrd volume and .csv plane lists
│   ├── MRHead_1.nrrd      # Primary 256x256x130 3D Slicer MRI reference dataset
│   └── sample_planes.csv  # 30 valid 3-point (triangle) and 4-point (trapezoid) planes
├── docs/                  # Documentation & Software Requirements Specification
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
python Ex0.py data/MRHead_1.nrrd data/sample_planes.csv data/demo_output.mp4
```

Sample output:
```text
[Ex1 Pipeline] Volume: (256, 256, 130) (uint16) | Planes: 30 | Backend: numba
[Ex1 Timing] Load: 196.01 ms | CSV: 17.39 ms | Projection: 6.92 ms (0.23 ms/frame, 4334.3 FPS) | Video: 1179.91 ms
[Ex1 Output] Wrote 30 frames to data\demo_output.mp4
```

---

## 5. Running Unit Tests & Benchmarks

### Run Unit Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Run $500 \times 500 \times 500$ Performance Benchmark
```bash
python tests/benchmark.py
```

---

## 6. Performance Benchmark & Experiment Summary

### 6.1 Benchmark on $500 \times 500 \times 500$ Volume (`tests/benchmark.py`)
The benchmark evaluates 120 planar slices (60 `TrianglePlane` slices and 60 oblique 3D `TrapezoidPlane` slices) across a $500 \times 500 \times 500$ `uint8` volume ($125,000,000$ voxels, $119.2\text{ MB}$) after warmup:

| Metric | Measured Result | Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Volume Size** | $500 \times 500 \times 500$ (`uint8`) | $500^3$ bytes | Verified |
| **Frames Projected** | $120$ frames ($50\%$ Triangle, $50\%$ Trapezoid) | Sequential plane list | Verified |
| **Total Projection Time** | $758.49\text{ ms}$ | $\le 4000\text{ ms}$ (for 30 FPS) | **PASSED** |
| **Average Time per Frame** | **$6.32\text{ ms / frame}$** | $< 10.0\text{ ms / frame}$ | **PASSED** |
| **Net Projection FPS** | **$158.21\text{ FPS}$** | Minimum $\ge 30\text{ FPS}$, Target $\ge 100\text{ FPS}$ | **PASSED (158+ FPS)** |

### 6.2 Demo Experiment (`data/MRHead_1.nrrd` & `data/sample_planes.csv`)
- **Dataset**: The 3D Slicer `MRHead` dataset (`data/MRHead_1.nrrd`), a $256 \times 256 \times 130$ sagittal T1 MRI scan stored as gzip-compressed 16-bit integer voxels (`min=0, max=279`, converted cleanly to `uint16`).
- **Trajectory**: `data/sample_planes.csv` defines 30 sequential planes: 15 axial/sagittal isosceles triangles sweeping across $z \in [20, 104]$ followed by 15 oblique isosceles trapezoids tilted across $(y, z)$.
- **Experimental Result**: `python Ex0.py data/MRHead_1.nrrd data/sample_planes.csv data/demo_output.mp4` projected all 30 frames in **$6.92\text{ ms}$ ($0.23\text{ ms/frame}$, $4,334.3\text{ FPS}$)** using the Numba parallel JIT backend and encoded `data/demo_output.mp4` ($224 \times 250$, $30\text{ FPS}$, H.264 `avc1` Constrained Baseline `yuv420p`, $289,282\text{ bytes}$).
