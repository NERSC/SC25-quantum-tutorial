# Running Infleqtion's logical qubit tutorials

Install the necessary python packages needed to run all the material by running `pip install -r requirements.txt` on the `requirements.txt` file in this directory.

1) Running the logical VQE ground state prep demo with `cudaq-solvers`:
    - Run the Jupyter notebook `infleqtion_logical_aim.ipynb` using the environment containing the installed packages from the `requirements.txt`. If a GPU is available, the CUDA-Q GPU backend will be utilized for the Hamiltonian Variational Ansatz optimization. This notebook will walk through the creation of logically encoded quantum circuits with the [[4,2,2]] code and compare their performance against an unencoded, or plain physical version, for preparing the ground state of the minimal single-impurity Anderson model.

2) Running decoding for the many-hypercube quantum error correcting code using `cudaq-qec` and `qLDPC`:
    - From this directory, run `python3 mhc_qldpc_decoding.py` to execute the python script to run decoding of real experimental data from Infleqtion's Sqale QPU. In particular, the script will decode the [[16, 4, 4]] many-hypercube logical state preparation reported in <https://arxiv.org/abs/2509.13247>. It uses `cudaq-qec`'s GPU-accelerated belief propagation (BP) decoder, along with the `qLDPC` library to demonstrate a pipeline for decoding syndromes and retrieving logical state outcomes.

3) Running an atom rearrangement experiment on Infleqtion's Sqale QPU:
    - To run the associated notebook, users should first acquire a Superstaq API key. Instructions to do so can be found [here](https://superstaq.readthedocs.io/en/latest/get_started/credentials.html).
    - Next, run the Jupyter notebook `infleqtion_atom_rearrangement.ipynb` using the environment containing the installed packages from the `requirements.txt`, and inserting your API key in the specified cell. This tutorial notebook will showcase the atom rearrangement capability of Infleqtion's Sqale QPU, a feature leveraged in executing Infleqtion's logical qubit experiments. Users will be able to submit a 2D bitmap array encoding a specified pattern via the presence or lack of an atom in the Sqale QPU array. Atom array capture results will be sent to the email associated with the user's API key.
