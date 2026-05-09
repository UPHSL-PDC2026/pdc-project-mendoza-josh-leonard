# PDC Finals Project — Team Brackie
**Parallel and Distributed Computing | UPHSL 2026**

---

## Project Overview

This project implements and compares three approaches to data processing on the **Bureau of Customs (BOC) 2019 import dataset**, a real-world government dataset containing 100,000 import transaction records with fields such as country of origin, goods description, dutiable value in PHP, duties paid, and exchange rates.

The same three analytics tasks are performed across all three approaches:

| Task | Description |
|---|---|
| **Filter** | Extract only records where `countryorigin_iso3 == 'CHN'` (China-origin imports) |
| **Aggregate** | Compute the average `dutiablevaluephp` (dutiable value in PHP) per country |
| **Sort** | Sort all records by `dutiablevaluephp` in descending order |

### The three approaches

**1. Sequential**: All tasks run one after another on a single process using pandas. This serves as the baseline for performance comparison.

**2. Parallel (MPI via mpi4py)**: Uses the Master-Worker pattern across 5 MPI processes. The master splits the dataset into 4 equal chunks and distributes them to worker processes. Each worker processes its chunk simultaneously, then sends results back to the master for merging.

**3. Distributed (Apache Spark via PySpark)**: Uses Spark in `local[*]` mode, which distributes data partitions across all available CPU cores automatically. Spark's lazy evaluation and DAG-based execution engine handle parallelism internally without manual data splitting.

After all three sections run, the program prints a side-by-side performance comparison table and a fault tolerance and scalability discussion.

---

## Tools and Technologies

| Tool | Version | Purpose |
|---|---|---|
| **Python** | 3.11.x | Runtime — PySpark 3.5.x requires Python 3.11 or lower |
| **pandas** | 3.0.x | Data loading and manipulation for sequential and MPI sections |
| **mpi4py** | 4.x | Python bindings for MPI — enables multi-process parallel execution |
| **PySpark** | 3.5.1 | Python API for Apache Spark — distributed data processing |
| **Apache Spark** | 3.5.1 | Distributed computation engine (bundled with PySpark) |
| **Java (JDK)** | 1.8 (Java 8) | Required by Spark's JVM runtime |
| **setuptools** | — | Restores `distutils` removed in Python 3.12+, required by PySpark internals |
| **mpiexec** | — | MPI process launcher, included with Microsoft MPI or OpenMPI |

### Windows-specific dependencies

| Dependency | Location | Purpose |
|---|---|---|
| `winutils.exe` | `C:\hadoop\bin\` | Required by Spark on Windows for file system operations |
| `hadoop.dll` | `C:\hadoop\bin\` | Required alongside winutils |

---

## Project Structure

```
PDC-Finals-TeamBrackie/
├── PDC-Fin-Project-TeamBrackie.py      # Main script (run via mpiexec)
├── PDC-Fin-Project-TeamBrackie.ipynb   # Jupyter notebook with source, output, table, and report
├── performancetable.png                # Screenshot of the performance comparison output
└── README.md                           # This file
```

> **Note:** `boc_lite_2019.csv` is not included in this repository (670 MB, exceeds GitHub's file size limit). The dataset must be placed locally at `C:\Users\<your_username>\Downloads\boc_lite_2019.csv` before running.

---

## Instructions for Running the Project

### Prerequisites

**1. Python 3.11**
PySpark 3.5.x does not support Python 3.12 or 3.13. Make sure Python 3.11 is installed.
```
py -0   # lists all installed Python versions
```

**2. Create a virtual environment with Python 3.11**
```bash
py -3.11 -m venv .venv
```

**3. Install dependencies**
```bash
.venv\Scripts\pip install pandas mpi4py pyspark==3.5.1 setuptools
```

**4. Install Microsoft MPI** (if not already installed)
Download from: https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi

**5. Set up winutils for Spark on Windows**

Download `winutils.exe` and `hadoop.dll` for Hadoop 3.3.5 and place them in `C:\hadoop\bin\`:
```powershell
New-Item -ItemType Directory -Force C:\hadoop\bin
New-Item -ItemType Directory -Force C:\tmp
# Download winutils.exe and hadoop.dll from:
# https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.5/bin
```

**6. Place the dataset**

Copy `boc_lite_2019.csv` to:
```
C:\Users\<your_username>\Downloads\boc_lite_2019.csv
```
Then update `CSV_PATH` in the script if your path is different:
```python
CSV_PATH = r'C:\Users\<your_username>\Downloads\boc_lite_2019.csv'
```

---

### Running the script

From the project directory, run:
```bash
mpiexec -n 5 .venv\Scripts\python.exe PDC-Fin-Project-TeamBrackie.py 2>$null
```

- `-n 5` — launches 5 processes: 1 master (rank 0) + 4 workers (ranks 1–4)
- `2>$null` — suppresses Java/Spark warning messages from the terminal output

Expected output sections:
```
DataFrame created in X.XXXX seconds.

=== Sequential Processing ===
=== Parallel Processing ===
=== Distributed Processing (PySpark) ===
=== Performance Comparison ===
=== Fault Tolerance & Scalability Discussion ===
```

---

### Running the notebook

Open `PDC-Fin-Project-TeamBrackie.ipynb` in Jupyter:
```bash
.venv\Scripts\jupyter notebook
```

Run the cells in order:
1. **Cell 1** — Source code (display only, do not run directly)
2. **Cell 2** — Runs the script via `subprocess` and captures output
3. **Cell 3** — Parses output and generates the performance comparison table
