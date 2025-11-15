from __future__ import annotations

try:
    import cudaq
    import cudaq_solvers as solvers
    import matplotlib.pyplot as plt
except ImportError:
    print("Installing required packages...")
    #pip install -r requirements.txt
    print("Installed `cudaq`, `cudaq-solvers`, and `matplotlib` packages.")
    print("You may need to restart the kernel to import newly installed packages.")
    import cudaq
    import cudaq_solvers as solvers
    import matplotlib.pyplot as plt

import os
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

if cudaq.num_available_gpus() == 0:
    cudaq.set_target("qpp-cpu", option="fp64")
else:
    cudaq.set_target("nvidia", option="fp64")

def ansatz(n_qubits: int) -> cudaq.Kernel:
    # Create a CUDA-Q parameterized kernel
    paramterized_ansatz, variational_angles = cudaq.make_kernel(list)
    qubits = paramterized_ansatz.qalloc(n_qubits)

    # Using |+> as the initial state:
    paramterized_ansatz.h(qubits[0])
    paramterized_ansatz.cx(qubits[0], qubits[1])

    paramterized_ansatz.rx(variational_angles[0], qubits[0])
    paramterized_ansatz.cx(qubits[0], qubits[1])
    paramterized_ansatz.rz(variational_angles[1], qubits[1])
    paramterized_ansatz.cx(qubits[0], qubits[1])
    return paramterized_ansatz


def run_logical_vqe(cudaq_hamiltonian: cudaq.SpinOperator) -> tuple[float, list[float]]:
    # Set seed for easier reproduction
    rng = np.random.default_rng(42)

    # Initial angles for the optimizer
    init_angles = rng.random(2) * 1e-1

    # Obtain CUDA-Q Ansatz
    num_qubits = cudaq_hamiltonian.qubit_count
    variational_kernel = ansatz(num_qubits)

    # Perform VQE optimization
    energy, params, _ = solvers.vqe(
        variational_kernel,
        cudaq_hamiltonian,
        init_angles,
        optimizer=minimize,
        method="SLSQP",
        tol=1e-10,
    )
    return energy, params

cudaq.register_operation("meas_id", np.identity(2))

def aim_physical_circuit(
    angles: list[float], basis: str, *, ignore_meas_id: bool = False
) -> cudaq.Kernel:
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)

    # Bell state prep
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])

    # Rx Gate
    kernel.rx(angles[0], qubits[0])

    # ZZ rotation
    kernel.cx(qubits[0], qubits[1])
    kernel.rz(angles[1], qubits[1])
    kernel.cx(qubits[0], qubits[1])

    if basis == "z_basis":
        if not ignore_meas_id:
            kernel.for_loop(
                start=0,
                stop=2,
                function=lambda q_idx: getattr(kernel, "meas_id")(qubits[q_idx]),  # noqa: B009
            )
        kernel.mz(qubits)
    elif basis == "x_basis":
        kernel.h(qubits)
        if not ignore_meas_id:
            kernel.for_loop(
                start=0,
                stop=2,
                function=lambda q_idx: getattr(kernel, "meas_id")(qubits[q_idx]),  # noqa: B009
            )
        kernel.mz(qubits)
    else:
        raise ValueError("Unsupported basis provided:", basis)
    return kernel


def aim_logical_circuit(
    angles: list[float], basis: str, *, ignore_meas_id: bool = False
) -> cudaq.Kernel:
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(6)

    kernel.for_loop(start=0, stop=3, function=lambda idx: kernel.h(qubits[idx]))
    kernel.cx(qubits[1], qubits[4])
    kernel.cx(qubits[2], qubits[3])
    kernel.cx(qubits[0], qubits[1])
    kernel.cx(qubits[0], qubits[3])

    # Rx teleportation
    kernel.rx(angles[0], qubits[0])

    kernel.cx(qubits[0], qubits[1])
    kernel.cx(qubits[0], qubits[3])
    kernel.h(qubits[0])

    if basis == "z_basis":
        if not ignore_meas_id:
            kernel.for_loop(
                start=0,
                stop=5,
                function=lambda idx: getattr(kernel, "meas_id")(qubits[idx]),  # noqa: B009
            )
        kernel.mz(qubits)
    elif basis == "x_basis":
        # ZZ rotation and teleportation
        kernel.cx(qubits[3], qubits[5])
        kernel.cx(qubits[2], qubits[5])
        kernel.rz(angles[1], qubits[5])
        kernel.cx(qubits[1], qubits[5])
        kernel.cx(qubits[4], qubits[5])
        kernel.for_loop(start=1, stop=5, function=lambda idx: kernel.h(qubits[idx]))
        if not ignore_meas_id:
            kernel.for_loop(
                start=0,
                stop=6,
                function=lambda idx: getattr(kernel, "meas_id")(qubits[idx]),  # noqa: B009
            )
        kernel.mz(qubits)
    else:
        raise ValueError("Unsupported basis provided:", basis)
    return kernel

def generate_circuit_set(ignore_meas_id: bool = False) -> object:
    u_vals = [1, 5, 9]
    v_vals = [-9, -1, 7]
    circuit_dict = {}
    for u in u_vals:
        for v in v_vals:
            qubit_hamiltonian = (
                0.25 * u * cudaq.spin.z(0) * cudaq.spin.z(1)
                - 0.25 * u
                + v * cudaq.spin.x(0)
                + v * cudaq.spin.x(1)
            )
            _, opt_params = run_logical_vqe(qubit_hamiltonian)
            angles = [float(angle) for angle in opt_params]
            print(f"Computed optimal angles={angles} for U={u}, V={v}")

            tmp_physical_dict = {}
            tmp_logical_dict = {}
            for basis in ("z_basis", "x_basis"):
                tmp_physical_dict[basis] = aim_physical_circuit(
                    angles, basis, ignore_meas_id=ignore_meas_id
                )
                tmp_logical_dict[basis] = aim_logical_circuit(
                    angles, basis, ignore_meas_id=ignore_meas_id
                )

            circuit_dict[f"{u}:{v}"] = {
                "physical": tmp_physical_dict,
                "logical": tmp_logical_dict,
            }
    print("\nFinished building optimized circuits!")
    return circuit_dict

sim_circuit_dict = generate_circuit_set()
circuit_layers = sim_circuit_dict.keys()


def _num_qubits(counts: Mapping[str, float]) -> int:
    for key in counts:
        if key.isdecimal():
            return len(key)
    return 0


def process_counts(
    counts: Mapping[str, float],
    data_qubits: Sequence[int],
    flag_qubits: Sequence[int] = (),
) -> dict[str, float]:
    new_data: dict[str, float] = {}
    for key, val in counts.items():
        if not all(key[i] == "0" for i in flag_qubits):
            continue

        new_key = "".join(key[i] for i in data_qubits)

        if not set("01").issuperset(new_key):
            continue

        new_data.setdefault(new_key, 0)
        new_data[new_key] += val

    return new_data


def decode(counts: Mapping[str, float]) -> dict[str, float]:
    """Decode physical counts into logical counts. Should be called after `process_counts`."""
    if not counts:
        return {}

    num_qubits = _num_qubits(counts)
    assert num_qubits % 4 == 0

    physical_to_logical = {
        "0000": "00",
        "1111": "00",
        "0011": "01",
        "1100": "01",
        "0101": "10",
        "1010": "10",
        "0110": "11",
        "1001": "11",
    }

    new_data: dict[str, float] = {}
    for key, val in counts.items():
        physical_keys = [key[i : i + 4] for i in range(0, num_qubits, 4)]
        logical_keys = [physical_to_logical.get(physical_key) for physical_key in physical_keys]
        if None not in logical_keys:
            new_key = "".join(logical_keys)
            new_data.setdefault(new_key, 0)
            new_data[new_key] += val

    return new_data


def ev_x(counts: Mapping[str, float]) -> float:
    ev = 0.0

    for k, val in counts.items():
        ev += val * ((-1) ** int(k[0]) + (-1) ** int(k[1]))

    total = sum(counts.values())
    ev /= total
    return ev


def ev_xx(counts: Mapping[str, float]) -> float:
    ev = 0.0

    for k, val in counts.items():
        ev += val * (-1) ** k.count("1")

    total = sum(counts.values())
    ev /= total
    return ev


def ev_zz(counts: Mapping[str, float]) -> float:
    ev = 0.0

    for k, val in counts.items():
        ev += val * (-1) ** k.count("1")

    total = sum(counts.values())
    ev /= total
    return ev


def aim_logical_energies(
    data_ordering: object, counts_list: Sequence[dict[str, float]]
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    counts_data = {
        data_ordering[i]: decode(
            process_counts(
                counts,
                data_qubits=[1, 2, 3, 4],
                flag_qubits=[0, 5],
            )
        )
        for i, counts in enumerate(counts_list)
    }
    return _aim_energies(counts_data)


def aim_physical_energies(
    data_ordering: object, counts_list: Sequence[dict[str, float]]
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    counts_data = {
        data_ordering[i]: process_counts(
            counts,
            data_qubits=[0, 1],
        )
        for i, counts in enumerate(counts_list)
    }
    return _aim_energies(counts_data)


def _aim_energies(
    counts_data: Mapping[tuple[int, int, str], dict[str, float]],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    evxs: dict[tuple[int, int], float] = {}
    evxxs: dict[tuple[int, int], float] = {}
    evzzs: dict[tuple[int, int], float] = {}
    totals: dict[tuple[int, int], float] = {}

    for key, counts in counts_data.items():
        h_params, basis = key
        key_a, key_b = h_params.split(":")
        u, v = int(key_a), int(key_b)
        if basis.startswith("x"):
            evxs[u, v] = ev_x(counts)
            evxxs[u, v] = ev_xx(counts)
        else:
            evzzs[u, v] = ev_zz(counts)

        totals.setdefault((u, v), 0)
        totals[u, v] += sum(counts.values())

    energies = {}
    uncertainties = {}
    for u, v in evxs.keys() & evzzs.keys():
        string_key = f"{u}:{v}"
        energies[string_key] = u * (evzzs[u, v] - 1) / 4 + v * evxs[u, v]

        uncertainty_xx = 2 * v**2 * (1 + evxxs[u, v]) - u * v * evxs[u, v] / 2
        uncertainty_zz = u**2 * (1 - evzzs[u, v]) / 2

        uncertainties[string_key] = np.sqrt(
            (uncertainty_zz + uncertainty_xx - energies[string_key] ** 2) / (totals[u, v] / 2)
        )

    return energies, uncertainties


def _get_energy_diff(
    bf_energies: dict[str, float],
    physical_energies: dict[str, float],
    logical_energies: dict[str, float],
) -> tuple[list[float], list[float]]:
    physical_energy_diff = []
    logical_energy_diff = []

    # Data ordering following `bf_energies` keys
    for layer in bf_energies.keys():
        physical_sim_energy = physical_energies[layer]
        logical_sim_energy = logical_energies[layer]
        true_energy = bf_energies[layer]
        u, v = layer.split(":")
        print(f"Layer=({u}, {v}) has brute-force energy of: {true_energy}")
        print(f"Physical circuit of layer=({u}, {v}) got an energy of: {physical_sim_energy}")
        print(f"Logical circuit of layer=({u}, {v}) got an energy of: {logical_sim_energy}")
        print("-" * 72)

        if logical_sim_energy < physical_sim_energy:
            print("Logical circuit achieved the lower energy!")
        else:
            print("Physical circuit achieved the lower energy")
        print("-" * 72, "\n")

        physical_energy_diff.append(
            -1 * (true_energy - physical_sim_energy)
        )  # Multiply by -1 since negative energies
        logical_energy_diff.append(-1 * (true_energy - logical_sim_energy))
    return physical_energy_diff, logical_energy_diff

def submit_aim_circuits(
    circuit_dict: object,
    *,
    folder_path: str = "future_aim_results",
    shots_count: int = 1000,
    noise_model: cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime.NoiseModel | None = None,
    run_async: bool = False,
) -> dict[str, list[dict[str, int]]] | None:
    if run_async:
        os.makedirs(folder_path, exist_ok=True)
    else:
        aim_results = {"physical": [], "logical": []}

    for layer in circuit_dict.keys():
        if run_async:
            print(f"Posting circuits associated with layer=('{layer}')")
        else:
            print(f"Running circuits associated with layer=('{layer}')")

        for basis in ("z_basis", "x_basis"):
            if run_async:
                u, v = layer.split(":")

                tmp_physical_results = cudaq.sample_async(
                    circuit_dict[layer]["physical"][basis], shots_count=shots_count
                )
                with open(
                    f"{folder_path}/physical_{basis}_job_u={u}_v={v}_result.txt", "w"
                ) as file:
                    file.write(str(tmp_physical_results))

                tmp_logical_results = cudaq.sample_async(
                    circuit_dict[layer]["logical"][basis], shots_count=shots_count
                )
                with open(f"{folder_path}/logical_{basis}_job_u={u}_v={v}_result.txt", "w") as file:
                    file.write(str(tmp_logical_results))
            else:
                tmp_physical_results = cudaq.sample(
                    circuit_dict[layer]["physical"][basis],
                    shots_count=shots_count,
                    noise_model=noise_model,
                )
                tmp_logical_results = cudaq.sample(
                    circuit_dict[layer]["logical"][basis],
                    shots_count=shots_count,
                    noise_model=noise_model,
                )
                aim_results["physical"].append({k: v for k, v in tmp_physical_results.items()})
                aim_results["logical"].append({k: v for k, v in tmp_logical_results.items()})
    if not run_async:
        print("\nCompleted all circuit sampling!")
        return aim_results
    else:
        print("\nAll circuits submitted for async sampling!")

def _get_async_results(
    layers: object, *, folder_path: str = "future_aim_results"
) -> dict[str, list[dict[str, int]]]:
    aim_results = {"physical": [], "logical": []}
    for layer in layers:
        print(f"Retrieving all circuits counts associated with layer=('{layer}')")
        u, v = layer.split(":")
        for basis in ("z_basis", "x_basis"):
            with open(f"{folder_path}/physical_{basis}_job_u={u}_v={v}_result.txt") as file:
                tmp_physical_results = cudaq.AsyncSampleResult(str(file.read()))
            physical_counts = tmp_physical_results.get()

            with open(f"{folder_path}/logical_{basis}_job_u={u}_v={v}_result.txt") as file:
                tmp_logical_results = cudaq.AsyncSampleResult(str(file.read()))
            logical_counts = tmp_logical_results.get()

            aim_results["physical"].append({k: v for k, v in physical_counts.items()})
            aim_results["logical"].append({k: v for k, v in logical_counts.items()})

    print("\nObtained all circuit samples!")
    return aim_results

cudaq.reset_target()
cudaq.set_target("density-matrix-cpu")

def get_device_noise(
    depolar_prob_1q: float,
    depolar_prob_2q: float,
    *,
    readout_error_prob: float | None = None,
    custom_gates: list[str] | None = None,
) -> cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime.NoiseModel:
    noise = cudaq.NoiseModel()
    depolar_noise = cudaq.DepolarizationChannel(depolar_prob_1q)

    noisy_ops = ["z", "s", "x", "h", "rx", "rz"]
    for op in noisy_ops:
        noise.add_all_qubit_channel(op, depolar_noise)

    if custom_gates:
        custom_depolar_channel = cudaq.DepolarizationChannel(depolar_prob_1q)
        for op in custom_gates:
            noise.add_all_qubit_channel(op, custom_depolar_channel)

    # Two qubit depolarization error
    p_0 = 1 - depolar_prob_2q
    p_1 = np.sqrt((1 - p_0**2) / 3)

    k0 = np.array(
        [[p_0, 0.0, 0.0, 0.0], [0.0, p_0, 0.0, 0.0], [0.0, 0.0, p_0, 0.0], [0.0, 0.0, 0.0, p_0]],
        dtype=np.complex128,
    )
    k1 = np.array(
        [[0.0, 0.0, p_1, 0.0], [0.0, 0.0, 0.0, p_1], [p_1, 0.0, 0.0, 0.0], [0.0, p_1, 0.0, 0.0]],
        dtype=np.complex128,
    )
    k2 = np.array(
        [
            [0.0, 0.0, -1j * p_1, 0.0],
            [0.0, 0.0, 0.0, -1j * p_1],
            [1j * p_1, 0.0, 0.0, 0.0],
            [0.0, 1j * p_1, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    k3 = np.array(
        [[p_1, 0.0, 0.0, 0.0], [0.0, p_1, 0.0, 0.0], [0.0, 0.0, -p_1, 0.0], [0.0, 0.0, 0.0, -p_1]],
        dtype=np.complex128,
    )
    kraus_channel = cudaq.KrausChannel([k0, k1, k2, k3])

    noise.add_all_qubit_channel("cz", kraus_channel)
    noise.add_all_qubit_channel("cx", kraus_channel)

    if readout_error_prob is not None:
        # Readout error modeled with a Bit flip channel on identity before measurement
        bit_flip = cudaq.BitFlipChannel(readout_error_prob)
        noise.add_all_qubit_channel("meas_id", bit_flip)
    return noise

# Example parameters that can model execution on hardware at the high, simulation, level:
# Take single-qubit gate depolarization rate: ~0.2% or better (fidelity ≥99.8%)
# Take two-qubit gate depolarization rate: ~1-2% (fidelity ~98-99%)
cudaq_noise_model = get_device_noise(0.002, 0.02, readout_error_prob=0.02)

aim_sim_data = submit_aim_circuits(sim_circuit_dict, noise_model=cudaq_noise_model)

data_ordering = []
for key in circuit_layers:
    for basis in ("z_basis", "x_basis"):
        data_ordering.append((key, basis))

sim_physical_energies, sim_physical_uncertainties = aim_physical_energies(
    data_ordering, aim_sim_data["physical"]
)

sim_logical_energies, sim_logical_uncertainties = aim_logical_energies(
    data_ordering, aim_sim_data["logical"]
)

bf_energies = {
    "1:-9": -18.251736027394713,
    "1:-1": -2.265564437074638,
    "1:7": -14.252231964940428,
    "5:-9": -19.293350575766127,
    "5:-1": -3.608495283014149,
    "5:7": -15.305692796870582,
    "9:-9": -20.39007993367173,
    "9:-1": -5.260398644698076,
    "9:7": -16.429650912487233,
}

sim_physical_energy_diff, sim_logical_energy_diff = _get_energy_diff(
    bf_energies, sim_physical_energies, sim_logical_energies
)

fig, ax = plt.subplots(figsize=(11, 7), dpi=150)

layer_labels = [(int(key.split(":")[0]), int(key.split(":")[1])) for key in bf_energies.keys()]
plot_labels = [str(item) for item in layer_labels]

plt.errorbar(
    plot_labels,
    sim_physical_energy_diff,
    yerr=sim_physical_uncertainties.values(),
    ecolor=(20 / 255.0, 26 / 255.0, 94 / 255.0),
    color=(20 / 255.0, 26 / 255.0, 94 / 255.0),
    capsize=4,
    elinewidth=1.5,
    fmt="o",
    markersize=8,
    markeredgewidth=1,
    label="Physical",
)

plt.errorbar(
    plot_labels,
    sim_logical_energy_diff,
    yerr=sim_logical_uncertainties.values(),
    color=(0, 177 / 255.0, 152 / 255.0),
    ecolor=(0, 177 / 255.0, 152 / 255.0),
    capsize=4,
    elinewidth=1.5,
    fmt="o",
    markersize=8,
    markeredgewidth=1,
    label="Logical",
)

ax.set_xlabel("Hamiltonian Parameters (U, V)", fontsize=18)
ax.set_ylabel("Energy above true ground state (in eV)", fontsize=18)
ax.set_title("CUDA-Q AIM Circuits Simulation (lower is better)", fontsize=20)
ax.legend(loc="upper right", fontsize=18.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax.axhline(y=0, color="black", linestyle="--", linewidth=2)
plt.ylim(
    top=max(sim_physical_energy_diff) + max(sim_physical_uncertainties.values()) + 0.2, bottom=-0.2
)
plt.tight_layout()
plt.show()
