# Parallel Data Processing System (Single-Node)

## Project Overview
This project evaluates the performance and efficiency of parallel computing versus sequential processing using a large-scale dataset from the Bureau of Customs (BOC) 2019. The core objective is to demonstrate how distributed systems can optimize data-intensive tasks such as filtering, aggregation, and sorting. The system utilizes a Master-Worker architecture to distribute data chunks across multiple processors, performing operations concurrently to achieve a significant speedup in execution time.

## Tools and Technologies Used
- Python: The primary programming language used for data manipulation and script execution.
- Pandas: Utilized for high-performance data structures and data analysis tools.
- mpi4py (Message Passing Interface for Python): The library used to implement distributed computing and manage inter-process communication.
- MS-MPI / mpiexec: The underlying execution environment required to run parallel processes on a Windows-based system.
- Jupyter Notebook: Used for staging the source code and documenting the output.

## Instructions for Running the Project

### 1. Prerequisites
Ensure you have the following installed on your system to support the distributed environment:
* **Python 3.x**
* **Microsoft MPI (MS-MPI)**: Required for the `mpiexec` execution environment on Windows.
* **Required Python libraries**:
    ```bash
    pip install pandas mpi4py
    ```
### 2. File Preparation
Before running the script, perform the following setup:
* **Dataset**: Place the `boc_lite_2019.csv` file in the directory specified within the source code.
* **Path Configuration**: Open the `.py` script (e.g., `PDC-Mid-Proj1-MendozaJoshLeonard.py`) and update the **local file paths** to match your specific folder structure for both the dataset and the Python interpreter.

### 3. Execution
To execute the parallel implementation, use the `mpiexec` command in your terminal. For this specific project, **5 processes** are recommended (1 Master node and 4 Worker nodes) to balance the workload:

```bash
mpiexec -n 5 python PDC-Mid-Proj1-MendozaJoshLeonard.py
```

### 4. Viewing Results
Once executed, the script will output the following in the terminal:
* **Sequential Processing**: Time benchmarks for each task performed on a single core.
* **Parallel Status**: Real-time updates from each worker node as they complete their assigned data chunks.
* **Performance Summary**: A final comparative analysis showing the total time and efficiency gains between the two methods.
