# SC25 Quantum Tutorial

Repository contains all the tutorial materials for the SC25 tutorial: [**Accelerated Quantum Supercomputing in Action: A Hands-on Tutorial on Scalable Hybrid Workflows**](https://sc25.conference-program.com/presentation/?id=tut162&sess=sess272).

This tutorial is a collaboration between [NVIDIA](https://www.nvidia.com/), [NERSC](https://www.nersc.gov/), and [Infleqtion](https://www.infleqtion.com/).

## Description

Accelerated quantum supercomputing (AQSC) tightly integrates quantum computing with classical accelerated supercomputing via low-latency interconnects. This is crucial for hybrid quantum-classical workflows, enabling scalable quantum algorithms, real-time quantum error correction (QEC), and fast feedback control.

In this tutorial, participants will gain hands-on experience by building hybrid applications using the Python API of `CUDA-Q`, NVIDIA’s open-source development platform that unifies QPU, CPU, and GPU compute. The primary focus is on scalable hybrid algorithms like the generative quantum eigensolver (GQE), emphasizing AI integration and parallelization.

Live demonstrations on NERSC’s Perlmutter supercomputer and Infleqtion’s Sqale neutral-atom QPU will showcase the value of GPU-accelerated quantum applications and QEC. Practical examples in `CUDA-Q` will include GPU-accelerated decoders on Perlmutter and the demonstration of logical qubits using VQE for a material science application.

For advanced participants, notebooks will also cover algorithms such QAOA-GPT and Auxiliary Field Quantum Monte-Carlo (AFMQC). For additional examples, please visit the [applications section in CUDA-Q docs](https://nvidia.github.io/cuda-quantum/latest/using/applications.html). 

Participants will leave with practical skills in building hybrid quantum-classical applications, an understanding of performance-critical AQSC components, and familiarity with emerging techniques in scalable quantum algorithm design and QEC.

Dedicated compute on Perlmutter and Infleqtion QPU will be provided during the tutorial.

## Tutorial Schedule

The tutorial runs from **1:30pm - 5:00pm CST**. The tentative agenda follows:

* **1:30pm - 2:30pm:** Overview of Accelerated Quantum Supercomputing, including a hands-on VQE example with `CUDA-Q`.

* **2:30pm - 3:00pm:** Generative Quantum Eigensolver (GQE)

* **3:00pm - 3:30pm:** *Break*

* **3:30pm - 4:00pm:** GQE (continued) + examples from AFQMC

* **4:00pm - 5:00pm:** Infleqtion Sqale Platform

## Resources

Tutorial notebooks and other resources for each session are found in the directories of this repository.

Additional material can be found at:

* **CUDA-Q Docs:** <https://github.com/NVIDIA/cuda-quantum>

* **CUDA-QX Docs:** <https://github.com/NVIDIA/cudaqx>

* **CUDA-Q Academic:** <https://github.com/NVIDIA/cuda-q-academic>
