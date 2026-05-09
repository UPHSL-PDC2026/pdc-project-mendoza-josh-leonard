# Code Explainer: mpi4py and PySpark in PDC-Fin-Project-TeamBrackie

---

## Table of Contents
1. [How mpi4py Works](#1-how-mpi4py-works)
2. [How PySpark Works](#2-how-pyspark-works)
3. [Project Overview](#3-project-overview)
4. [Code Walkthrough — Imports and Setup](#4-code-walkthrough--imports-and-setup)
5. [Code Walkthrough — Sequential](#5-code-walkthrough--sequential)
6. [Code Walkthrough — Parallel (MPI)](#6-code-walkthrough--parallel-mpi)
7. [Code Walkthrough — Distributed (PySpark)](#7-code-walkthrough--distributed-pyspark)
8. [Code Walkthrough — Comparison and Main](#8-code-walkthrough--comparison-and-main)

---

## 1. How mpi4py Works

**MPI** stands for **Message Passing Interface**. It is a standard that defines how multiple processes can communicate with each other — even across different machines on a network. `mpi4py` is the Python library that implements this standard.

### The core idea

When you run a script with `mpiexec -n 5 python script.py`, MPI does **not** run one Python process. It runs **5 completely separate Python processes at the same time**, each with its own memory. These processes cannot share variables directly — the only way they can exchange data is by explicitly **sending** and **receiving** messages.

```
Process 0 (Master)  ──send(chunk)──►  Process 1 (Worker)
                    ──send(chunk)──►  Process 2 (Worker)
                    ──send(chunk)──►  Process 3 (Worker)
                    ──send(chunk)──►  Process 4 (Worker)

Process 1  ──send(result)──►  Process 0
Process 2  ──send(result)──►  Process 0
Process 3  ──send(result)──►  Process 0
Process 4  ──send(result)──►  Process 0
```

### Key concepts

| Concept | What it means |
|---|---|
| **Communicator** (`COMM_WORLD`) | The group that contains all running processes |
| **Rank** | The unique ID number of each process (0, 1, 2, ...) |
| **Size** | The total number of processes running |
| **send / recv** | How processes pass data to each other |
| **Barrier** | A synchronization point — all processes must reach it before any can continue |
| **Master-Worker pattern** | Rank 0 is the "boss" that distributes work; all other ranks are "workers" that do the actual processing |

### Why it's fast

Because all workers run **simultaneously**. If you have 4 workers and a dataset of 100,000 rows, each worker only processes 25,000 rows at the same time. The total work is done in roughly 1/4 of the time (minus communication overhead).

---

## 2. How PySpark Works

**Apache Spark** is a distributed data processing engine originally built for large-scale data on clusters of machines. **PySpark** is the Python API for Spark.

### The core idea

Spark works by breaking a dataset into **partitions** and distributing those partitions across available CPU cores (or machines in a real cluster). It then runs operations on all partitions **in parallel** using a **DAG (Directed Acyclic Graph)** of transformations.

```
Original DataFrame
        │
        ▼
  ┌─────────────┐
  │  Partition 1│──► Core 1
  │  Partition 2│──► Core 2
  │  Partition 3│──► Core 3
  │  Partition 4│──► Core 4
  └─────────────┘
        │
        ▼
   Merged Result
```

### Lazy evaluation

This is the most important concept in Spark. When you write:

```python
d_filtered = sdf.filter(F.col('countryorigin_iso3') == 'CHN')
```

**Nothing actually happens yet.** Spark just records the instruction in a plan. The actual computation only runs when you call an **action** — something that needs a real result, like `.count()` or `.toPandas()`. This allows Spark to optimize the entire pipeline before executing it.

| Type | Examples | What it does |
|---|---|---|
| **Transformation** | `.filter()`, `.groupBy()`, `.orderBy()` | Defines what to do — lazy, no execution yet |
| **Action** | `.count()`, `.toPandas()`, `.show()` | Triggers actual execution |

### Key concepts

| Concept | What it means |
|---|---|
| **SparkSession** | The entry point — like opening a connection to the Spark engine |
| **DataFrame (sdf)** | A distributed table, similar to a pandas DataFrame but split across cores |
| **local[\*]** | Run Spark on the local machine using all available CPU cores |
| **cache()** | Store the DataFrame in memory so repeated operations don't re-read from disk |
| **Fault tolerance** | If a partition is lost (e.g., a worker crashes), Spark recomputes it from the original data using the recorded DAG — no full restart needed |

### Why it's different from MPI

MPI requires you to manually split data, send it, process it, and collect it. Spark handles all of that automatically — you just write what you want to do, and Spark figures out how to distribute it.

---

## 3. Project Overview

The project runs the **same three analytics tasks** using three different approaches and compares their execution times:

| Task | What it does |
|---|---|
| **Filter** | Keep only rows where `countryorigin_iso3 == 'CHN'` |
| **Aggregate** | Calculate the average `dutiablevaluephp` per country |
| **Sort** | Sort all rows by `dutiablevaluephp` in descending order |

The dataset is the **BOC 2019 lite import data**, loaded as 100,000 rows.

---

## 4. Code Walkthrough — Imports and Setup

```python
# mpiexec -n 5 .venv\Scripts\python.exe PDC-Fin-Project-TeamBrackie.py 2>$null
```
This comment is the terminal command used to run the script. `-n 5` means 5 processes total (1 master + 4 workers). `2>$null` hides Java/Spark warning messages from the terminal output.

---

```python
import pandas as pd
import time
from mpi4py import MPI
```
- `pandas` — used for loading and manipulating the CSV data
- `time` — used to measure how long each task takes
- `MPI` from `mpi4py` — gives access to all MPI communication functions

---

```python
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
```
These three lines run **in every process** the moment the script starts.

- `MPI.COMM_WORLD` — gets the communicator that contains all 5 processes
- `comm.Get_rank()` — each process gets its own unique number: master gets `0`, workers get `1`, `2`, `3`, `4`
- `comm.Get_size()` — every process learns the total count, which is `5`

This is how the script knows whether it is the master or a worker — by checking `rank`.

---

```python
CSV_PATH = r'C:\Users\Josh\Downloads\boc_lite_2019.csv'
NROWS    = 100_000
```
Constants for the dataset path and how many rows to load. Using constants at the top makes it easy to change without hunting through the code.

---

```python
start_time = time.time()
```
Records the wall-clock time at the very start. Used at the end to calculate total elapsed time.

---

```python
s_timepertask = []
p_timepertask = []
d_timepertask = []
```
Three lists that store the time each task takes for Sequential, Parallel, and Distributed respectively. These are used later to build the comparison table.

---

## 5. Code Walkthrough — Sequential

```python
def sequential(df):
```
Takes the already-loaded pandas DataFrame as input. Only called by rank 0 (the master).

---

```python
    t0 = time.time()
    s_filtered = df[df['countryorigin_iso3'] == 'CHN']
    elapsed = time.time() - t0
    s_timepertask.append(elapsed)
```
- `t0 = time.time()` — snapshot the time before the task starts
- `df[df['countryorigin_iso3'] == 'CHN']` — pandas boolean indexing: creates a new DataFrame containing only rows where the country of origin column equals `'CHN'`
- `time.time() - t0` — calculates how many seconds the task took
- `.append(elapsed)` — saves that time to the list for later comparison

The same pattern (`t0` → operation → `elapsed` → `append`) is repeated for every task in every section.

---

```python
    s_agg = df.groupby('countryorigin_iso3').agg({'dutiablevaluephp': 'mean'}).reset_index()
```
- `.groupby('countryorigin_iso3')` — groups all rows by their country code
- `.agg({'dutiablevaluephp': 'mean'})` — for each group, calculate the mean (average) of the `dutiablevaluephp` column
- `.reset_index()` — converts the group labels back into a regular column instead of an index

---

```python
    s_sorted = df.sort_values(by='dutiablevaluephp', ascending=False)
```
Sorts the entire DataFrame by `dutiablevaluephp` from highest to lowest. `ascending=False` means descending order.

---

## 6. Code Walkthrough — Parallel (MPI)

```python
def parallel(df=None):
```
`df=None` is the default because workers call this function without passing a DataFrame — they receive their data from the master via MPI instead.

---

### Master side (`if rank == 0`)

```python
        workers = size - 1
        total_rows = len(df)
        rows_per_worker = total_rows // workers
```
- `size - 1` — excludes the master from the worker count (master coordinates, doesn't process data)
- `total_rows // workers` — integer division to split rows evenly. With 100,000 rows and 4 workers, each gets 25,000 rows.

---

```python
        for i in range(1, size):
            s = (i - 1) * rows_per_worker
            e = s + rows_per_worker if i < size - 1 else total_rows
            comm.send(df.iloc[s:e].copy(), dest=i, tag=0)
```
This loop sends a chunk of the DataFrame to each worker.

- `range(1, size)` — loops through worker ranks: 1, 2, 3, 4
- `s` and `e` — calculate the start and end row index for each worker's chunk
- `if i < size - 1 else total_rows` — the last worker gets any remaining rows (handles cases where rows don't divide evenly)
- `df.iloc[s:e]` — slices the DataFrame by row position
- `.copy()` — makes a clean copy so the slice doesn't share memory with the original
- `comm.send(..., dest=i, tag=0)` — sends the chunk to worker `i`. `tag=0` is a label that identifies this message as the "data distribution" message (as opposed to task results)

---

```python
        parts = [comm.recv(source=i, tag=1) for i in range(1, size)]
        p_filtered = pd.concat(parts, ignore_index=True)
```
- `comm.recv(source=i, tag=1)` — waits to receive the filtered result from each worker. `tag=1` identifies this as a Task 1 (filter) result
- The list comprehension collects all 4 results into a list
- `pd.concat(parts, ignore_index=True)` — merges all 4 filtered chunks back into one DataFrame. `ignore_index=True` resets the row numbers so they don't overlap

---

```python
        parts = [comm.recv(source=i, tag=2) for i in range(1, size)]
        p_agg = pd.concat(parts, ignore_index=True) \
                  .groupby('countryorigin_iso3').agg({'dutiablevaluephp': 'mean'}).reset_index()
```
Each worker computed a local average for its chunk. But those local averages are not the true global average — they need to be re-aggregated. The master collects all partial aggregates (`tag=2`), concatenates them, then runs `groupby().agg('mean')` again to get the correct global average per country.

---

```python
        parts = [comm.recv(source=i, tag=3) for i in range(1, size)]
        p_sorted = pd.concat(parts, ignore_index=True) \
                     .sort_values(by='dutiablevaluephp', ascending=False)
```
Each worker sorted its own chunk locally. The master collects all sorted chunks (`tag=3`), concatenates them into one DataFrame, then does a final global sort. This is why the parallel sort is sometimes slower than sequential — the master has to do extra work to merge and re-sort.

---

### Worker side (`else`)

```python
    else:
        local_df = comm.recv(source=0, tag=0)
```
Workers wait here until the master sends them their chunk. `source=0` means "receive from rank 0 (master)". `tag=0` matches the tag the master used when sending.

---

```python
        local_filtered = local_df[local_df['countryorigin_iso3'] == 'CHN']
        comm.send(local_filtered, dest=0, tag=1)
        print(f'Processor {rank} has done Task 1')
```
- The worker filters its local chunk (same logic as sequential, just on a smaller slice)
- `comm.send(..., dest=0, tag=1)` — sends the result back to the master. `tag=1` identifies it as a Task 1 result
- The print confirms the worker finished (these appear in the terminal before the master's output)

The same pattern repeats for Task 2 (aggregate, `tag=2`) and Task 3 (sort, `tag=3`).

---

## 7. Code Walkthrough — Distributed (PySpark)

```python
def distributed(df):
```
Takes the pandas DataFrame already loaded by the master. Only rank 0 calls this — Spark manages its own internal parallelism independently of MPI.

---

```python
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
```
- `SparkSession` — the main entry point to Spark. Everything goes through this object.
- `functions as F` — Spark's built-in column functions (like `F.col()`, `F.mean()`). Imported as `F` to keep code short.

---

```python
    venv_python = os.path.abspath(sys.executable)
    os.environ.setdefault('HADOOP_HOME', r'C:\hadoop')
    os.environ['PYSPARK_PYTHON'] = venv_python
    os.environ['PYSPARK_DRIVER_PYTHON'] = venv_python
```
Windows-specific setup required before starting Spark:

- `os.path.abspath(sys.executable)` — gets the full absolute path to the Python executable currently running. This is needed because `mpiexec` can change the working directory, making relative paths fail.
- `HADOOP_HOME` — tells Spark where to find `winutils.exe`, a small utility Spark needs on Windows to perform file operations. Without it, Spark crashes.
- `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` — tells Spark's JVM which Python executable to use when spawning worker processes. Without this, Spark might find the wrong Python version on the system.

---

```python
    spark = (
        SparkSession.builder
        .appName('PDC-Final-TeamBrackie')
        .master('local[*]')
        .config('spark.driver.memory', '2g')
        .config('spark.sql.shuffle.partitions', str((size - 1) * 2))
        .config('spark.log.structuredLogging.enabled', 'false')
        .getOrCreate()
    )
```
This builds and starts the Spark session using a **builder pattern** (chaining `.config()` calls):

- `.appName(...)` — a label for this Spark job, visible in Spark's web UI
- `.master('local[*]')` — run Spark locally using **all available CPU cores** (`*` means all). On a real cluster this would be something like `spark://master:7077`
- `.config('spark.driver.memory', '2g')` — gives the Spark driver process 2 GB of RAM
- `.config('spark.sql.shuffle.partitions', ...)` — controls how many partitions are created during shuffle operations (like groupBy). Set to `(size - 1) * 2` = 8, which matches the number of MPI workers × 2 for better parallelism
- `.config('spark.log.structuredLogging.enabled', 'false')` — disables a newer logging format that causes issues on some setups
- `.getOrCreate()` — starts the session, or reuses an existing one if already running

---

```python
    spark.sparkContext.setLogLevel('ERROR')
    log4j = spark.sparkContext._jvm.org.apache.log4j
    log4j.Logger.getLogger('org.apache.spark.util.ShutdownHookManager').setLevel(log4j.Level.OFF)
    log4j.Logger.getLogger('org.apache.spark.SparkContext').setLevel(log4j.Level.OFF)
```
- `setLogLevel('ERROR')` — tells Spark to only print error messages, hiding the many INFO and WARN messages it normally produces
- The `log4j` lines access Spark's internal Java logging system directly through `_jvm` (the bridge between Python and the JVM) and silence two specific loggers that produce noisy cleanup messages on Windows shutdown

---

```python
    sdf = spark.createDataFrame(df).cache()
    sdf.count()
```
- `spark.createDataFrame(df)` — converts the pandas DataFrame into a Spark DataFrame. Spark splits it into partitions automatically across available cores. This approach is used instead of reading the CSV directly because Spark's CSV reader doesn't support `latin-1` encoding.
- `.cache()` — tells Spark to keep this DataFrame in memory after it's first computed, so the three tasks don't re-process the data from scratch each time
- `sdf.count()` — this is an **action** that forces Spark to actually load and cache the data. Without this, the cache wouldn't be populated until the first task runs.

---

```python
    d_filtered = sdf.filter(F.col('countryorigin_iso3') == 'CHN')
    d_filtered.count()   # action – triggers execution
```
- `sdf.filter(F.col('countryorigin_iso3') == 'CHN')` — a **transformation**. `F.col('countryorigin_iso3')` refers to the column by name. This line just records the filter plan — nothing runs yet.
- `d_filtered.count()` — the **action** that triggers Spark to actually execute the filter across all partitions in parallel. The result (row count) is discarded; we only call it to force execution so the timing is accurate.

---

```python
    print(d_filtered.limit(5).toPandas().head(5))
```
- `.limit(5)` — tells Spark to only return 5 rows (avoids transferring the entire filtered result)
- `.toPandas()` — collects the Spark DataFrame result back to the driver as a pandas DataFrame. This is an action.
- `.head(5)` — shows the first 5 rows of the pandas result

---

```python
    d_agg = sdf.groupBy('countryorigin_iso3') \
               .agg(F.mean('dutiablevaluephp').alias('dutiablevaluephp'))
    d_agg.count()
```
- `sdf.groupBy('countryorigin_iso3')` — groups rows by country code across all partitions
- `.agg(F.mean('dutiablevaluephp').alias('dutiablevaluephp'))` — computes the mean of `dutiablevaluephp` for each group. `.alias(...)` renames the result column to keep the same column name as the sequential version
- `d_agg.count()` — action that triggers the groupBy + aggregation to actually execute

---

```python
    d_sorted = sdf.orderBy(F.col('dutiablevaluephp').desc())
    d_sorted.count()
```
- `sdf.orderBy(F.col('dutiablevaluephp').desc())` — sorts the entire distributed DataFrame by `dutiablevaluephp` in descending order. `.desc()` specifies descending direction.
- `d_sorted.count()` — triggers execution

---

```python
    spark.stop()
```
Shuts down the Spark session and releases all resources (memory, threads, JVM). Always important to call this at the end to avoid resource leaks.

---

## 8. Code Walkthrough — Comparison and Main

```python
def print_comparison():
    ...
    for i, label in enumerate(labels):
        s = f"{s_timepertask[i]:.4f}s" if i < len(s_timepertask) else 'N/A'
        p = f"{p_timepertask[i]:.4f}s" if i < len(p_timepertask) else 'N/A'
        d = f"{d_timepertask[i]:.4f}s" if i < len(d_timepertask) else 'N/A'
```
Reads from the three timing lists that were populated during each section. `:.4f` formats the number to 4 decimal places. The `if i < len(...)` guard prevents an index error if a section was skipped.

---

```python
    if s_t and p_t:
        print(f'\nMPI Speedup   vs Sequential : {s_t / p_t:.2f}x')
```
Speedup is calculated as `sequential_time / parallel_time`. A value greater than 1.0x means the parallel version was faster. A value less than 1.0x means it was slower (which can happen when communication overhead exceeds the benefit of parallelism on small datasets).

---

```python
def main():
    if rank == 0:
        df = pd.read_csv(CSV_PATH, encoding='latin-1', nrows=NROWS)
        ...
        sequential(df)

    comm.Barrier()
```
- Only rank 0 loads the data and runs sequential — workers have nothing to do here
- `comm.Barrier()` — **all 5 processes must reach this line before any of them can continue**. This ensures the sequential section is fully complete before the parallel section begins. Without this, workers might start the parallel section before the master is ready to send them data.

---

```python
    if rank == 0:
        df = pd.read_csv(CSV_PATH, encoding='latin-1', nrows=NROWS)
        parallel(df)
    else:
        parallel()

    comm.Barrier()
```
- The master loads a fresh copy of the data and calls `parallel(df)` with it
- Workers call `parallel()` with no argument — they receive their data from the master inside the function via `comm.recv()`
- The second `comm.Barrier()` ensures all MPI processes finish the parallel section before rank 0 starts the Spark session

---

```python
    if rank == 0:
        df = pd.read_csv(CSV_PATH, encoding='latin-1', nrows=NROWS)
        distributed(df)
        print_comparison()
        print_discussion()
        print(f'{time.time() - start_time:.4f} seconds in total.')
```
Only rank 0 runs Spark — workers have nothing to do here since Spark manages its own internal parallelism. After Spark finishes, the comparison table and discussion are printed, followed by the total wall-clock time since the script started.
