# SC25-quantum-tutorial
Repository contains all the tutorial materials for the SC25 workshop [Accelerated Quantum Supercomputing in Action: A Hands-on Tutorial on Scalable Hybrid Algorithms and Quantum Error Correction ](https://sc24.conference-program.com/presentation/?id=tut167&sess=sess407), a collaboration with NERSC, NVIDIA, and Infleqtion. 

## Description

Accelerated quantum supercomputing (AQSC) tightly integrates quantum computing with classical accelerated supercomputing via low-latency interconnects. This is crucial for hybrid quantum-classical workflows, enabling scalable quantum algorithms, real-time quantum error correction (QEC), and fast feedback control.

In this tutorial, participants will gain hands-on experience by building hybrid applications using the Python API of CUDA-Q, NVIDIA’s open-source development platform that unifies QPU, CPU, and GPU compute. The primary focus is on scalable hybrid algorithms like the generative quantum eigensolver (GQE), emphasizing AI integration and parallelization.

Live demonstrations on NERSC’s Perlmutter supercomputer and Infleqtion’s Sqale neutral-atom QPU will showcase the value of GPU-accelerated quantum applications and QEC. Practical examples in CUDA-Q will include GPU-accelerated decoders on Perlmutter and the demonstration of logical qubits using VQE for a material science application. For advanced participants, notebooks will also cover algorithms such as contextual machine learning (CML), QAOA-GPT, and Auxiliary Field Quantum Monte-Carlo (AFMQC).

Participants will leave with practical skills in building hybrid quantum-classical applications, an understanding of performance-critical AQSC components, and familiarity with emerging techniques in scalable quantum algorithm design and QEC.

Dedicated compute on Perlmutter and Infleqtion QPU will be provided during the tutorial.

## Tutorial Schedule

The tutorial runs from 1:30am - 5:00pm EST.  There are four sessions separated by breaks.  The tentative agenda for each session follows:

* **8:30-10:00am:** Overview of methods of accelerating quantum simulation with GPUs including a hands-on QAOA example with CUDA-Q 
* 10:00-10:30: Break
* **10:30-noon:** Live demo local install of CUDA-Q followed by an introduction to large scale clusters, how to navigate and use them 
* noon-1:30pm: Lunch break
* **1:30-3:00pm:** Hands-on Example: Quantum Chemistry and Nuclear Physics examples at NERSC and a industry use case of simulating Hamiltonians of molecules with 30,000 terms
* 3:00-3:30pm: Break 
* **3:30-5:00pm:** Finish the industry use case example, run a Quantum Resevoir Computing example with QuEra, and conclude the session

## Resources
The slides for all the sessions are collated in the file [quantum-accelerated-supercomputing-sc24.pdf](quantum-accelerated-supercomputing-sc24.pdf). Tutorial notebooks and other resources for each session are found in the directories of this repository.
