# mpiexec -n 5 .venv\Scripts\python.exe PDC-Fin-Project-TeamBrackie.py 2>$null

import pandas as pd
import time
from mpi4py import MPI

# ── MPI Setup ─────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

CSV_PATH = r'C:\Users\Josh\Downloads\boc_lite_2019.csv'
NROWS    = 100_000

start_time = time.time()

# Timing lists
s_timepertask = []
p_timepertask = []
d_timepertask = []


# ──────────────────────────────────────────────────────────────────────────────
# SEQUENTIAL
# ──────────────────────────────────────────────────────────────────────────────
def sequential(df):
    print('=== Sequential Processing ===\n')

    # Task 1 – Filter
    print('> Task 1: Filter Data\n')
    t0 = time.time()
    s_filtered = df[df['countryorigin_iso3'] == 'CHN']
    elapsed = time.time() - t0
    s_timepertask.append(elapsed)
    print(s_filtered.head(5))
    print(f'\n> Task done in {elapsed:.4f} seconds.\n')
    print('==============')

    # Task 2 – Aggregate
    print('> Task 2: Aggregate Data\n')
    t0 = time.time()
    s_agg = df.groupby('countryorigin_iso3').agg({'dutiablevaluephp': 'mean'}).reset_index()
    elapsed = time.time() - t0
    s_timepertask.append(elapsed)
    print(s_agg)
    print(f'\n> Task done in {elapsed:.4f} seconds.\n')
    print('==============')

    # Task 3 – Sort
    print('> Task 3: Sort Data\n')
    t0 = time.time()
    s_sorted = df.sort_values(by='dutiablevaluephp', ascending=False)
    elapsed = time.time() - t0
    s_timepertask.append(elapsed)
    print(s_sorted.head(5))
    print(f'\n> Task done in {elapsed:.4f} seconds.\n')
    print('==============\n')

    print(f'Sequential Processing Time Total: {sum(s_timepertask):.4f} seconds.')


# ──────────────────────────────────────────────────────────────────────────────
# PARALLEL  (MPI)
# ──────────────────────────────────────────────────────────────────────────────
def parallel(df=None):
    if rank == 0:
        print('\n=== Parallel Processing ===\n')

        workers = size - 1
        total_rows = len(df)
        rows_per_worker = total_rows // workers

        # Distribute chunks
        for i in range(1, size):
            s = (i - 1) * rows_per_worker
            e = s + rows_per_worker if i < size - 1 else total_rows
            comm.send(df.iloc[s:e].copy(), dest=i, tag=0)

        # Task 1 – Filter
        print('> Task 1: Filter Data\n')
        t0 = time.time()
        parts = [comm.recv(source=i, tag=1) for i in range(1, size)]
        p_filtered = pd.concat(parts, ignore_index=True)
        elapsed = time.time() - t0
        p_timepertask.append(elapsed)
        print(p_filtered.head(5))
        print(f'\n> Task done in {elapsed:.4f} seconds.\n')
        print('==============')

        # Task 2 – Aggregate
        print('> Task 2: Aggregate Data\n')
        t0 = time.time()
        parts = [comm.recv(source=i, tag=2) for i in range(1, size)]
        p_agg = pd.concat(parts, ignore_index=True) \
                  .groupby('countryorigin_iso3').agg({'dutiablevaluephp': 'mean'}).reset_index()
        elapsed = time.time() - t0
        p_timepertask.append(elapsed)
        print(p_agg)
        print(f'\n> Task done in {elapsed:.4f} seconds.\n')
        print('==============')

        # Task 3 – Sort
        print('> Task 3: Sort Data\n')
        t0 = time.time()
        parts = [comm.recv(source=i, tag=3) for i in range(1, size)]
        p_sorted = pd.concat(parts, ignore_index=True) \
                     .sort_values(by='dutiablevaluephp', ascending=False)
        elapsed = time.time() - t0
        p_timepertask.append(elapsed)
        print(p_sorted.head(5))
        print(f'\n> Task done in {elapsed:.4f} seconds.\n')
        print('==============\n')

        print(f'Parallel Processing Time Total: {sum(p_timepertask):.4f} seconds.')

    else:
        local_df = comm.recv(source=0, tag=0)

        local_filtered = local_df[local_df['countryorigin_iso3'] == 'CHN']
        comm.send(local_filtered, dest=0, tag=1)
        print(f'Processor {rank} has done Task 1')

        local_agg = local_df.groupby('countryorigin_iso3') \
                             .agg({'dutiablevaluephp': 'mean'}).reset_index()
        comm.send(local_agg, dest=0, tag=2)
        print(f'Processor {rank} has done Task 2')

        local_sorted = local_df.sort_values(by='dutiablevaluephp', ascending=False)
        comm.send(local_sorted, dest=0, tag=3)
        print(f'Processor {rank} has done Task 3')


# ──────────────────────────────────────────────────────────────────────────────
# DISTRIBUTED  (PySpark)
# ──────────────────────────────────────────────────────────────────────────────
def distributed(df):
    """
    PySpark distributed processing.
    pandas DataFrame is passed in (already loaded with latin-1 encoding)
    and converted to a Spark DataFrame to avoid Spark's encoding limitation.
    """
    print('\n=== Distributed Processing (PySpark) ===\n')

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
    except ImportError:
        print('[SKIP] PySpark not installed.  Run: pip install pyspark\n')
        return

    import os, sys, warnings
    warnings.filterwarnings('ignore')

    # Required on Windows: point Spark at winutils.exe and the correct Python.
    # Use an absolute path so mpiexec doesn't break the relative path resolution.
    venv_python = os.path.abspath(sys.executable)
    os.environ.setdefault('HADOOP_HOME', r'C:\hadoop')
    os.environ['PYSPARK_PYTHON'] = venv_python
    os.environ['PYSPARK_DRIVER_PYTHON'] = venv_python

    spark = (
        SparkSession.builder
        .appName('PDC-Final-TeamBrackie')
        .master('local[*]')
        .config('spark.driver.memory', '2g')
        .config('spark.sql.shuffle.partitions', str((size - 1) * 2))
        .config('spark.log.structuredLogging.enabled', 'false')
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel('ERROR')
    # Silence the noisy ShutdownHookManager cleanup error on Windows
    log4j = spark.sparkContext._jvm.org.apache.log4j
    log4j.Logger.getLogger('org.apache.spark.util.ShutdownHookManager').setLevel(log4j.Level.OFF)
    log4j.Logger.getLogger('org.apache.spark.SparkContext').setLevel(log4j.Level.OFF)

    print(f'Spark {spark.version}  |  Master: {spark.sparkContext.master}'
          f'  |  Cores: {spark.sparkContext.defaultParallelism}\n')

    # Convert pandas → Spark (bypasses latin-1 issue)
    t_load = time.time()
    sdf = spark.createDataFrame(df).cache()
    sdf.count()   # materialise cache
    print(f'DataFrame created in {time.time() - t_load:.4f} seconds.\n')

    # Task 1 – Filter
    print('> Task 1: Filter Data\n')
    t0 = time.time()
    d_filtered = sdf.filter(F.col('countryorigin_iso3') == 'CHN')
    d_filtered.count()   # action – triggers execution
    elapsed = time.time() - t0
    d_timepertask.append(elapsed)
    print(d_filtered.limit(5).toPandas().head(5))
    print(f'\n> Task done in {elapsed:.4f} seconds.\n')
    print('==============')

    # Task 2 – Aggregate
    print('> Task 2: Aggregate Data\n')
    t0 = time.time()
    d_agg = sdf.groupBy('countryorigin_iso3') \
               .agg(F.mean('dutiablevaluephp').alias('dutiablevaluephp'))
    d_agg.count()   # action
    elapsed = time.time() - t0
    d_timepertask.append(elapsed)
    print(d_agg.orderBy('countryorigin_iso3').toPandas())
    print(f'\n> Task done in {elapsed:.4f} seconds.\n')
    print('==============')

    # Task 3 – Sort
    print('> Task 3: Sort Data\n')
    t0 = time.time()
    d_sorted = sdf.orderBy(F.col('dutiablevaluephp').desc())
    d_sorted.count()   # action
    elapsed = time.time() - t0
    d_timepertask.append(elapsed)
    print(d_sorted.limit(5).toPandas().head(5))
    print(f'\n> Task done in {elapsed:.4f} seconds.\n')
    print('==============\n')

    print(f'Distributed Processing Time Total: {sum(d_timepertask):.4f} seconds.')

    spark.stop()


# ──────────────────────────────────────────────────────────────────────────────
# COMPARISON + DISCUSSION
# ──────────────────────────────────────────────────────────────────────────────
def print_comparison():
    labels = ['Filter', 'Aggregate', 'Sort']

    print('\n=== Performance Comparison ===\n')
    print(f"{'Task':<12} {'Sequential':>14} {'Parallel(MPI)':>15} {'Distributed(Spark)':>20}")
    print('-' * 62)

    for i, label in enumerate(labels):
        s = f"{s_timepertask[i]:.4f}s" if i < len(s_timepertask) else 'N/A'
        p = f"{p_timepertask[i]:.4f}s" if i < len(p_timepertask) else 'N/A'
        d = f"{d_timepertask[i]:.4f}s" if i < len(d_timepertask) else 'N/A'
        print(f'{label:<12} {s:>14} {p:>15} {d:>20}')

    print('-' * 62)
    s_t = sum(s_timepertask)
    p_t = sum(p_timepertask)
    d_t = sum(d_timepertask)
    print(f"{'TOTAL':<12} {s_t:>13.4f}s {p_t:>14.4f}s {d_t:>19.4f}s")

    if s_t and p_t:
        print(f'\nMPI Speedup   vs Sequential : {s_t / p_t:.2f}x')
    if s_t and d_t:
        print(f'Spark Speedup vs Sequential : {s_t / d_t:.2f}x')
    if p_t and d_t:
        print(f'Spark Speedup vs MPI        : {p_t / d_t:.2f}x')


def print_discussion():
    print("""
=== Fault Tolerance & Scalability Discussion ===

Sequential
  - No fault tolerance. A crash loses all progress.
  - Bounded by a single CPU core; does not scale.

Parallel - MPI (mpi4py)
  - No built-in recovery. If one worker crashes, the whole job fails.
  - Mitigation: checkpoint intermediate results to disk and restart.
  - Scales well on multi-core machines and HPC clusters.
  - Communication overhead grows with process count (Amdahl's Law).

Distributed - PySpark
  - Built-in fault tolerance via RDD lineage (DAG).
    If a partition is lost, Spark recomputes only that partition
    from its parent — no full restart needed.
  - Checkpointing (rdd.checkpoint() or writing to Parquet/HDFS)
    cuts recomputation cost for long pipelines.
  - Designed for horizontal scaling: add worker nodes and Spark
    redistributes partitions automatically.
  - Best for large-scale ETL and cloud pipelines (EMR, Databricks).
  - Note: JVM startup overhead makes it slower than MPI on small
    datasets. Advantage shows at millions of rows on a real cluster.
""")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    if rank == 0:
        df = pd.read_csv(CSV_PATH, encoding='latin-1', nrows=NROWS)
        read_time = time.time() - start_time
        print(f'DataFrame created in {read_time:.4f} seconds.\n\n')

        sequential(df)

    comm.Barrier()

    if rank == 0:
        df = pd.read_csv(CSV_PATH, encoding='latin-1', nrows=NROWS)
        parallel(df)
    else:
        parallel()

    comm.Barrier()

    if rank == 0:
        df = pd.read_csv(CSV_PATH, encoding='latin-1', nrows=NROWS)
        distributed(df)

        print_comparison()
        print_discussion()

        print(f'{time.time() - start_time:.4f} seconds in total.')

main()
