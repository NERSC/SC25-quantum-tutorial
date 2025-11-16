# How to run GQE?

Run the requirements.txt file using `pip install -r requirements.txt`. 

## Algorithm Description
See the [slide deck](https://github.com/NERSC/SC25-quantum-tutorial/blob/main/Hour2/GQE/gqe_slides.pdf) to learn about this algorithm.

## Script Descriptions

There are three main scripts in this folder, each with a specific purpose.

### 1. `gqe_simple.py`

* **Purpose:** This is a foundational script designed to illustrate the basic workflow of the GQE algorithm.
* **Best for:** New users who want to understand the core logic and simple building blocks of GQE without the complexity of a specific scientific problem.

### 2. `gqe_h2.py`

* **Purpose:** A practical, hands-on example that uses GQE to compute the ground state energy of a hydrogen (H2) molecule.
* **Features:** This script is flexible and can be run in several configurations:
    * On a single GPU or a CPU
    * In parallel using multiple GPUs, can be run on multiple nodes using MPI.

### 3. `gqe_h2_Perlmutter.py`

* **Purpose:** An version of the H2 molecule script, specifically for running on the **Perlmutter supercomputer** at NERSC.
* **Features:** This script is built to demonstrates how to run GQE across **multiple GPUs and multiple nodes (MGMN)**.
* **Best for:** Users interested in learning how to scale this algorithm on Perlmutter.
