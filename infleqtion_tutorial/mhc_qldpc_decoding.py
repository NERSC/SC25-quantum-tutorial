# Copyright 2025 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING

import cudaq_qec as qec
import numpy as np
import qldpc

if TYPE_CHECKING:
    import numpy.typing as npt


def get_4_2_2_code() -> qldpc.codes.CSSCode:
    code = qldpc.codes.IcebergCode(2)

    # Permute qubit order to match Infleqtion's hardware experiments
    ops_z = code.get_logical_ops(qldpc.objects.Pauli.Z).take([0, 2, 3, 1], 1)
    ops_x = code.get_logical_ops(qldpc.objects.Pauli.X).take([0, 2, 3, 1], 1)
    code.set_logical_ops_xz(ops_x, ops_z)
    return code


def get_many_hypercube_code() -> qldpc.codes.CSSCode:
    code_422 = get_4_2_2_code()
    return qldpc.codes.CSSCode.concatenate(code_422, code_422)


def _decode_ancilla_flag(key: str) -> str | None:
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
    return physical_to_logical.get(key)


def compute_tvd(counts_1: dict[str, float], counts_2: dict[str, float]) -> float:
    """Computes the total variational distance (TVD) between two distributions."""
    p = _normalize(counts_1)
    q = _normalize(counts_2)

    all_keys = set(p.keys()).union(q.keys())
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in all_keys)


def _normalize(counts: dict[str, float]) -> dict[str, float]:
    norm_factor = sum(counts.values())
    return {k: v / norm_factor for k, v in counts.items()}


def get_logical_counts(
    experiment_counts: Mapping[str, int],
    parity_check_mat: npt.NDArray[np.uint8],
    logical_z_ops: npt.NDArray[np.uint8],
) -> dict[str, float]:
    osd_method = 3
    osd_order = 2
    nv_dec_args = {
        "max_iterations": 1000,
        "use_sparsity": False,
        "use_osd": osd_method > 0,
        "osd_order": osd_order,
        "osd_method": osd_method,
        "bp_method": 0,
    }

    # Get the BP-OSD qLDPC decoder from `cudaq-qec`
    try:
        nv_gpu_decoder = qec.get_decoder("nv-qldpc-decoder", parity_check_mat, **nv_dec_args)
    except Exception:
        raise AttributeError(
            "The `nv-qldpc-decoder` is not available with your current CUDA-Q QEC installation."
        )

    def _decode_syndromes(
        bitstring: str, *, bp_converged_flags: list[bool] | None = None
    ) -> npt.NDArray[np.uint8] | None:
        data_bits = np.asarray(list(map(int, bitstring)), dtype=np.uint8)
        syndrome = parity_check_mat.dot(data_bits) % 2

        # Check if `syndrome` has detected an error, and if not, no corrections are needed
        if not syndrome.any():
            return data_bits

        # Next, handle a special case many-hypercube sydrome correction
        # (weight-4 error can arise during prep without error-correction gadgets)
        if syndrome[:4].all() and not syndrome[4:].any():
            data_bits[3::4] ^= 1
            return data_bits

        # Otherwise, use the decoder to determine correction:
        results = nv_gpu_decoder.decode(syndrome)

        if bp_converged_flags is not None:
            bp_converged_flags.append(bool(results.converged))

        dec_result = np.asarray(results.result, dtype=np.uint8).ravel()
        if dec_result.sum() == 1:
            corrected_bits = data_bits ^ dec_result
            return corrected_bits
        return None

    decoding_time = 0.0
    bp_converged_flags: list[bool] = []
    logical_counts: dict[str, float] = {}
    for bitstring, count in experiment_counts.items():
        logical_flag = bitstring[:4]
        data_bitstring = bitstring[4:]
        assert len(data_bitstring) == 16, "Measured data bitstring is not of 16 qubits"

        if _decode_ancilla_flag(logical_flag) is None:
            continue

        t0 = time.time()
        corrected_bits = _decode_syndromes(data_bitstring, bp_converged_flags=bp_converged_flags)
        t1 = time.time()
        decoding_time += t1 - t0

        if corrected_bits is None:
            continue

        # Compute the logical Z parity to infer the logical state
        logical_bits = logical_z_ops.dot(corrected_bits) % 2
        logical_bitstring = "".join(map(str, logical_bits))
        logical_counts.setdefault(logical_bitstring, 0)
        logical_counts[logical_bitstring] += count

    print(f"Total syndrome decoding time: {1e3 * decoding_time:.3f} ms")
    return logical_counts


if __name__ == "__main__":
    # Get info on [[4,2,2]]-many-hypercube code from the `qLDPC` package:
    mhc_code = get_many_hypercube_code()
    logical_z_ops = np.asarray(mhc_code.get_logical_ops(qldpc.objects.Pauli.Z), dtype=np.uint8)
    parity_check_mat = np.asarray(mhc_code.matrix_z, dtype=np.uint8)

    print()
    mhc_state_preps = ("0000", "0111", "1110", "1111")
    for prep_string in mhc_state_preps:
        # Load experiment data run on Infleqtion's Sqale:
        with open(f"mhc_sqale_data/mhc_state_{prep_string}_prep_data_counts.json") as file:
            mhc_exp_data = json.load(file)

        print(f"Running decoding for logical |{prep_string}> state ...")
        logical_counts = get_logical_counts(mhc_exp_data, parity_check_mat, logical_z_ops)
        tvd = compute_tvd(logical_counts, {prep_string: 1})
        print(f"Obtained a TVD of {tvd:.5f} against ideal for logical |{prep_string}> state")
        print("--" * 32, "\n")
