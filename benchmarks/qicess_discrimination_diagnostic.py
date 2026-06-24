#!/usr/bin/env python3
"""
QICESS contact-Ising discrimination diagnostic (Steps 1-3).

Step 1: Hamiltonian sign / normalization audit
Step 2: Alanine dipeptide Ramachandran basin separation
Step 3: One principled fix from diagnosis, re-run on 14-protein held-out set
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pennylane as qml
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.ablation_study import (  # noqa: E402
    exact_diag_ground_state,
    evaluate_ranking,
    find_common_residues,
    score_ensemble_random,
)
from configs.benchmark_dataset import AUTOINHIBITED_BENCHMARK, get_autoinhibited_benchmark
from src.data.pdb_fetcher import (
    compute_contact_map,
    compute_phi_psi,
    fetch_pdb,
    parse_pdb_ca_coords,
)
from src.ensemble.conformational_sampler import generate_hybrid_ensemble
from src.quantum.exact_ising import IsingModel, exact_ground_state
from src.quantum.ising_vqe import MJ_AA_TO_IDX, MJ_POTENTIALS, build_ising_hamiltonian
from src.scoring.qicess_v2 import vqe_contact_agreement

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "qicess_diagnostic"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Original Papageorgiou core — first 14 autoinhibited targets
HELD_OUT_14 = AUTOINHIBITED_BENCHMARK[:14]


def contacts_to_qubit_bitstring(
    contact_map: np.ndarray, selected_contacts: List[Tuple[int, int, float]]
) -> str:
    bits = []
    n = contact_map.shape[0]
    for i, j, _ in selected_contacts:
        active = 1 if (i < n and j < n and contact_map[i, j] > 0) else 0
        bits.append(str(active))
    return "".join(bits)


def conformation_ising_energy(
    contact_map: np.ndarray,
    selected_contacts: List[Tuple[int, int, float]],
    model: IsingModel,
) -> float:
    bs = contacts_to_qubit_bitstring(contact_map, selected_contacts)
    return model.energy(bs)


def pennylane_hamiltonian_to_ising_model(H, n_qubits: int) -> IsingModel:
    return IsingModel.from_pennylane(H, n_qubits)


def audit_hamiltonian_sign(
    sequence: str,
    reference_coords: np.ndarray,
    fd_indices: List[int],
    im_indices: List[int],
) -> Dict:
    """Step 1 audit for one protein."""
    contact_map = compute_contact_map(reference_coords, threshold=8.0)
    interface_res = (fd_indices, im_indices) if fd_indices and im_indices else None
    built = build_ising_hamiltonian(sequence, contact_map, interface_res)
    if built is None or built[1] == 0:
        return {"status": "no_hamiltonian"}

    H, n_qubits, selected = built
    model = pennylane_hamiltonian_to_ising_model(H, n_qubits)
    exact = exact_diag_ground_state(H, n_qubits, selected)

    ref_bs = contacts_to_qubit_bitstring(contact_map, selected)
    ref_energy = model.energy(ref_bs)
    gs_bs = exact["ground_bitstring"]
    gs_energy = exact["ground_energy"]
    hamming_ref_gs = sum(a != b for a, b in zip(ref_bs, gs_bs))

    # Sign-flip test: negate all MJ couplings in selected contacts
    flipped_selected = [(i, j, -c) for i, j, c in selected]
    # Rebuild with negated J in contact map sense — flip sign on Hamiltonian coeffs
    H_flip, _, _ = build_ising_hamiltonian(sequence, contact_map, interface_res)
    flip_coeffs = [-float(c) for c in H_flip.coeffs]
    H_neg = qml.Hamiltonian(flip_coeffs, H_flip.ops)
    exact_neg = exact_diag_ground_state(H_neg, n_qubits, flipped_selected)

    return {
        "n_qubits": n_qubits,
        "n_selected_contacts": len(selected),
        "ref_bitstring": ref_bs,
        "ground_bitstring": gs_bs,
        "ref_energy": ref_energy,
        "ground_energy": gs_energy,
        "ref_matches_ground": ref_bs == gs_bs,
        "hamming_ref_vs_ground": hamming_ref_gs,
        "ref_agreement_with_ground": vqe_contact_agreement(
            contact_map, selected, gs_bs
        ),
        "negated_ground_bitstring": exact_neg["ground_bitstring"],
        "sign_flip_changes_ground": exact_neg["ground_bitstring"] != gs_bs,
    }


def rank_by_quantum_agreement_only(
    ensemble, sequence, reference_coords, fd_indices, im_indices, ground_bs, selected
) -> List[Dict]:
    scored = []
    for idx, conf in enumerate(ensemble):
        cm = compute_contact_map(conf["coords"], threshold=8.0)
        qa = vqe_contact_agreement(cm, selected, ground_bs)
        scored.append({**conf, "composite": qa, "quantum_agreement": qa, "original_idx": idx})
    scored.sort(key=lambda x: x["composite"], reverse=True)
    return scored


def rank_by_per_conformation_energy(
    ensemble, selected, model: IsingModel
) -> List[Dict]:
    """Rank by raw Ising energy of each conformation's contact pattern (lower=better)."""
    energies = []
    for conf in ensemble:
        cm = compute_contact_map(conf["coords"], threshold=8.0)
        e = conformation_ising_energy(cm, selected, model)
        energies.append(e)

    e_min, e_max = min(energies), max(energies)
    span = max(e_max - e_min, 1e-8)

    scored = []
    for idx, conf in enumerate(ensemble):
        # invert: lower energy → higher score
        score = 1.0 - (energies[idx] - e_min) / span
        scored.append(
            {
                **conf,
                "composite": score,
                "ising_energy": energies[idx],
                "original_idx": idx,
            }
        )
    scored.sort(key=lambda x: x["composite"], reverse=True)
    return scored


def rank_by_per_conformation_energy_per_contact(
    ensemble, selected, model: IsingModel
) -> List[Dict]:
    """Per-conformation Ising energy normalized by number of qubits (size correction)."""
    energies = []
    n_q = max(len(selected), 1)
    for conf in ensemble:
        cm = compute_contact_map(conf["coords"], threshold=8.0)
        e = conformation_ising_energy(cm, selected, model) / n_q
        energies.append(e)

    e_min, e_max = min(energies), max(energies)
    span = max(e_max - e_min, 1e-8)

    scored = []
    for idx, conf in enumerate(ensemble):
        score = 1.0 - (energies[idx] - e_min) / span
        scored.append({**conf, "composite": score, "original_idx": idx})
    scored.sort(key=lambda x: x["composite"], reverse=True)
    return scored


def build_alanine_dipeptide_coords(phi: float, psi: float) -> Tuple[np.ndarray, str]:
    """
    Idealized extended-chain alanine dipeptide (Ace-Ala-Nme) Cα trace in Å.
    phi/psi in degrees; used for basin discrimination only.
    """
    phi_r, psi_r = np.radians(phi), np.radians(psi)
    # Ace (res 0), Ala (res 1), Nme cap (res 2) — 3 Cα atoms
    ca = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [
                3.8 + 3.8 * np.cos(np.pi - phi_r),
                3.8 * np.sin(np.pi - phi_r),
                0.0,
            ],
        ]
    )
    # Apply psi rotation at residue 1
    pivot = ca[1]
    rel = ca[2] - pivot
    rot = _rotation_y(psi_r)
    ca[2] = pivot + rel @ rot.T
    seq = "AAA"
    return ca, seq


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def build_deca_alanine(phi: float, psi: float, n: int = 10) -> Tuple[np.ndarray, str]:
    """Deca-alanine with uniform (phi, psi) — proxy for Ramachandran basins at contact-map scale."""
    phi_r, psi_r = np.radians(phi), np.radians(psi)
    coords = [np.array([0.0, 0.0, 0.0])]
    for i in range(1, n):
        step = np.array([3.8, 0.0, 0.0])
        ang = phi_r if i % 2 == 1 else psi_r
        c, s = np.cos(ang), np.sin(ang)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        step = rot @ step
        coords.append(coords[-1] + step)
    return np.array(coords), "A" * n


def build_independent_qubit_hamiltonian(selected_contacts):
    """Step 3 fix: remove ZZ frustration — one Z term per contact qubit only."""
    import pennylane as qml

    coeffs, ops = [], []
    for qi, (_, _, ji) in enumerate(selected_contacts):
        coeffs.append(-ji / 2.0)
        ops.append(qml.PauliZ(qi))
    return qml.Hamiltonian(coeffs, ops)


def alanine_dipeptide_basin_test() -> Dict:
    """
    Step 2: Can contact-Ising separate clearly different Ramachandran basins?

    Literal Ace-Ala-Nme (3 Cα) yields zero contacts at 8 Å, so we use deca-alanine
    with uniform (phi, psi) per basin — the same contact-Ising pipeline, on basins
    with well-separated contact topology (not degenerate in energy by construction).
    """
    basins = {
        "C7eq": (-80.0, 80.0),
        "C7ax": (60.0, -60.0),
        "helical": (-57.0, -47.0),
        "extended": (-150.0, 150.0),
    }

    basin_coords = {}
    basin_contacts = {}
    for name, (phi, psi) in basins.items():
        coords, seq = build_deca_alanine(phi, psi)
        basin_coords[name] = coords
        basin_contacts[name] = compute_contact_map(coords, threshold=8.0)

    # 3-bead alanine dipeptide sanity check (expected: no contacts at 8 Å)
    di_coords, di_seq = build_alanine_dipeptide_coords(-80.0, 80.0)
    di_contacts = int(compute_contact_map(di_coords, threshold=8.0).sum() / 2)

    # Build Hamiltonian from C7eq reference (standard basin)
    ref_name = "C7eq"
    ref_coords, ref_seq = build_deca_alanine(*basins[ref_name])
    cm_ref = compute_contact_map(ref_coords, threshold=8.0)
    built = build_ising_hamiltonian(ref_seq, cm_ref, None)
    if built is None or built[1] == 0:
        return {
            "status": "no_hamiltonian",
            "note": "Too few contacts even for deca-alanine",
            "ala_dipeptide_contacts_at_8A": di_contacts,
        }

    H, n_qubits, selected = built
    model = pennylane_hamiltonian_to_ising_model(H, n_qubits)
    exact = exact_diag_ground_state(H, n_qubits, selected)
    gs_bs = exact["ground_bitstring"]

    rows = []
    for name in basins:
        cm = basin_contacts[name]
        bs = contacts_to_qubit_bitstring(cm, selected)
        e = model.energy(bs)
        qa = vqe_contact_agreement(cm, selected, gs_bs)
        rows.append(
            {
                "basin": name,
                "phi_psi": basins[name],
                "n_contacts": int(cm.sum() / 2),
                "bitstring": bs,
                "ising_energy": e,
                "agreement_with_ground": qa,
            }
        )

    # Pairwise energy gaps
    energies = {r["basin"]: r["ising_energy"] for r in rows}
    gap_eq_ax = energies["C7eq"] - energies["C7ax"]

    # Synthetic ensemble: perturb C7eq and C7ax with noise
    rng = np.random.default_rng(0)
    ensemble = []
    labels = []
    for _ in range(30):
        for name, (phi, psi) in [("C7eq", basins["C7eq"]), ("C7ax", basins["C7ax"])]:
            coords, _ = build_deca_alanine(
                phi + rng.normal(0, 5), psi + rng.normal(0, 5)
            )
            coords += rng.normal(0, 0.05, coords.shape)
            ensemble.append({"coords": coords, "label": name})
            labels.append(name)

    # Rank by per-conformation energy
    energies_ens = []
    for conf in ensemble:
        cm = compute_contact_map(conf["coords"], threshold=8.0)
        energies_ens.append(conformation_ising_energy(cm, selected, model))

    # C7eq should rank higher (lower energy) than C7ax on average
    eq_e = [energies_ens[i] for i, l in enumerate(labels) if l == "C7eq"]
    ax_e = [energies_ens[i] for i, l in enumerate(labels) if l == "C7ax"]
    _, p_mannwhitney = scipy_stats.mannwhitneyu(eq_e, ax_e, alternative="less")

    # Independent-qubit model (no ZZ): ref should match ground
    H_ind = build_independent_qubit_hamiltonian(selected)
    model_ind = pennylane_hamiltonian_to_ising_model(H_ind, n_qubits)
    exact_ind = exact_diag_ground_state(H_ind, n_qubits, selected)
    ref_bs = contacts_to_qubit_bitstring(cm_ref, selected)
    ind_rows = []
    for name in basins:
        cm = basin_contacts[name]
        bs = contacts_to_qubit_bitstring(cm, selected)
        ind_rows.append(
            {
                "basin": name,
                "ising_energy_independent": model_ind.energy(bs),
            }
        )
    ind_energies = {r["basin"]: r["ising_energy_independent"] for r in ind_rows}

    return {
        "n_qubits": n_qubits,
        "selected_contacts": len(selected),
        "ala_dipeptide_contacts_at_8A": di_contacts,
        "note": "3-bead Ace-Ala-Nme has 0 contacts at 8 Å; deca-alanine used for basin test",
        "basin_energies_full_hamiltonian": rows,
        "C7eq_minus_C7ax_energy": gap_eq_ax,
        "eq_mean_energy": float(np.mean(eq_e)),
        "ax_mean_energy": float(np.mean(ax_e)),
        "mannwhitney_eq_lower_p": float(p_mannwhitney),
        "separates_basins_full_h": float(np.mean(eq_e)) < float(np.mean(ax_e)),
        "independent_qubit_ref_matches_ground": ref_bs == exact_ind["ground_bitstring"],
        "basin_energies_independent_qubit": ind_rows,
        "separates_C7eq_vs_extended_independent": ind_energies["C7eq"]
        < ind_energies["extended"],
    }


def load_protein_data(target, max_ensemble: int = 65):
    if target.pdb_id_state1 == target.pdb_id_state2:
        return None
    pdb1 = fetch_pdb(target.pdb_id_state1)
    pdb2 = fetch_pdb(target.pdb_id_state2)
    if not pdb1 or not pdb2:
        return None

    s1 = parse_pdb_ca_coords(pdb1, chain=target.chain_state1) or parse_pdb_ca_coords(
        pdb1, chain=None
    )
    s2 = parse_pdb_ca_coords(pdb2, chain=target.chain_state2) or parse_pdb_ca_coords(
        pdb2, chain=None
    )
    if s1 is None or s2 is None or s1["n_residues"] > 1000:
        return None

    ci1, ci2, nc = find_common_residues(s1, s2)
    if nc < 20:
        return None

    fd_start, fd_end = target.fd_residues
    im_start, im_end = target.im_residues
    fd_idx = [i for i, r in enumerate(s1["residue_ids"]) if fd_start <= r <= fd_end]
    im_idx = [i for i, r in enumerate(s1["residue_ids"]) if im_start <= r <= im_end]
    if not fd_idx or not im_idx:
        n = s1["n_residues"]
        fd_idx, im_idx = list(range(n // 2, n)), list(range(0, n // 2))

    n_ens = min(max_ensemble, 80 if s1["n_residues"] < 400 else 50)
    phi_psi = compute_phi_psi(pdb1, chain=s1["chain"])
    ensemble = generate_hybrid_ensemble(
        s1["coords"],
        s1["sequence"],
        fd_indices=fd_idx,
        im_indices=im_idx,
        n_conformations=n_ens,
        seed=42,
        phi_psi=phi_psi,
    )
    for conf in ensemble:
        conf["phi_psi"] = phi_psi

    return {
        "target": target,
        "s1": s1,
        "s2": s2,
        "ci1": ci1,
        "ci2": ci2,
        "fd_idx": fd_idx,
        "im_idx": im_idx,
        "ensemble": ensemble,
    }


def run_14_protein_ranking(method_fn, label: str) -> Dict:
    """Run a ranking method on held-out 14 proteins; return pooled Spearman."""
    per_protein = []
    all_ranks, all_tms = [], []

    for target in HELD_OUT_14:
        data = load_protein_data(target)
        if data is None:
            continue

        s1, s2 = data["s1"], data["s2"]
        contact_map = compute_contact_map(s1["coords"], threshold=8.0)
        interface_res = (data["fd_idx"], data["im_idx"])
        built = build_ising_hamiltonian(
            s1["sequence"], contact_map, interface_res
        )
        if built is None or built[1] == 0:
            continue
        H, n_qubits, selected = built
        model = pennylane_hamiltonian_to_ising_model(H, n_qubits)
        exact = exact_diag_ground_state(H, n_qubits, selected)

        scored = method_fn(
            data["ensemble"],
            s1["sequence"],
            s1["coords"],
            data["fd_idx"],
            data["im_idx"],
            exact["ground_bitstring"],
            selected,
            model,
        )
        ev = evaluate_ranking(scored, s2["coords"], data["ci1"], data["ci2"], k=10)
        if ev is None:
            continue
        per_protein.append({"gene": target.gene_name, **ev})

        # Pool conformations across proteins (rank within protein only)
        tms = []
        for conf in scored:
            from src.metrics.structural_metrics import tm_score

            valid_ens = [i for i in data["ci1"] if i < len(conf["coords"])]
            valid_tgt = data["ci2"][: len(valid_ens)]
            if len(valid_ens) >= 10:
                tms.append(tm_score(s2["coords"][valid_tgt], conf["coords"][valid_ens]))
            else:
                tms.append(0.0)
        ranks = np.arange(len(tms))
        all_ranks.extend(ranks.tolist())
        all_tms.extend(tms)

    rhos = [p["spearman_rho"] for p in per_protein]
    rand_rhos = []
    for target in HELD_OUT_14:
        data = load_protein_data(target)
        if data is None:
            continue
        scored_rand = score_ensemble_random(data["ensemble"], seed=42)
        ev = evaluate_ranking(
            scored_rand, data["s2"]["coords"], data["ci1"], data["ci2"], k=10
        )
        if ev:
            rand_rhos.append(ev["spearman_rho"])

    mean_rho = float(np.mean(rhos)) if rhos else float("nan")
    mean_rand = float(np.mean(rand_rhos)) if rand_rhos else float("nan")

    # Paired Wilcoxon: exact vs random per protein
    exact_rhos = rhos
    if len(exact_rhos) >= 3 and len(rand_rhos) >= 3:
        n_pair = min(len(exact_rhos), len(rand_rhos))
        try:
            _, p_wilcoxon = scipy_stats.wilcoxon(
                np.array(exact_rhos[:n_pair]) - np.array(rand_rhos[:n_pair])
            )
        except Exception:
            p_wilcoxon = 1.0
    else:
        p_wilcoxon = 1.0

    return {
        "method": label,
        "n_proteins": len(per_protein),
        "mean_spearman_rho": mean_rho,
        "std_spearman_rho": float(np.std(rhos)) if rhos else float("nan"),
        "mean_random_spearman_rho": mean_rand,
        "wilcoxon_p_vs_random": float(p_wilcoxon),
        "per_protein": per_protein,
    }


def method_exact_agreement(ensemble, sequence, ref_coords, fd, im, gs_bs, selected, model):
    return rank_by_quantum_agreement_only(
        ensemble, sequence, ref_coords, fd, im, gs_bs, selected
    )


def method_per_conf_energy(ensemble, sequence, ref_coords, fd, im, gs_bs, selected, model):
    return rank_by_per_conformation_energy(ensemble, selected, model)


def method_per_conf_energy_norm(ensemble, sequence, ref_coords, fd, im, gs_bs, selected, model):
    return rank_by_per_conformation_energy_per_contact(ensemble, selected, model)


def collective_coordinate_score(
    coords: np.ndarray, fd_indices: List[int], im_indices: List[int]
) -> float:
    """
    Step 3 principled fix: domain packing angle + inter-domain COM distance.
    Lower hinge opening / tighter packing → higher score when normalized within ensemble.
    """
    if not fd_indices or not im_indices:
        return 0.5
    fd = coords[fd_indices]
    im = coords[im_indices]
    com_fd = fd.mean(axis=0)
    com_im = im.mean(axis=0)
    dist = float(np.linalg.norm(com_fd - com_im))

    # Domain axis vectors (N→C direction)
    axis_fd = fd[-1] - fd[0]
    axis_im = im[-1] - im[0]
    n_fd = np.linalg.norm(axis_fd)
    n_im = np.linalg.norm(axis_im)
    if n_fd < 1e-6 or n_im < 1e-6:
        cos_angle = 0.0
    else:
        cos_angle = float(
            np.dot(axis_fd / n_fd, axis_im / n_im)
        )
    # Combine: use raw features; normalization happens in rank function
    return dist, cos_angle


def rank_by_independent_qubit_energy(
    ensemble, sequence, ref_coords, fd, im, gs_bs, selected, model
) -> List[Dict]:
    """
    Step 3 principled fix from Step 1 diagnosis:
    ZZ cooperative terms drive the ground state away from the reference contact
    map (mean Hamming 5/16; ref energy >> ground energy). Remove ZZ frustration
    and rank by each conformation's own independent-qubit Ising energy.
    """
    H_ind = build_independent_qubit_hamiltonian(selected)
    model_ind = pennylane_hamiltonian_to_ising_model(H_ind, len(selected))
    return rank_by_per_conformation_energy(ensemble, selected, model_ind)


def run_step1_audit() -> Dict:
    logger.info("=" * 70)
    logger.info("STEP 1: Hamiltonian sign / normalization audit")
    logger.info("=" * 70)
    audits = []
    for target in HELD_OUT_14[:6]:  # representative subset
        pdb1 = fetch_pdb(target.pdb_id_state1)
        s1 = parse_pdb_ca_coords(pdb1, chain=target.chain_state1) or parse_pdb_ca_coords(
            pdb1, chain=None
        )
        if s1 is None:
            continue
        fd_start, fd_end = target.fd_residues
        im_start, im_end = target.im_residues
        fd_idx = [i for i, r in enumerate(s1["residue_ids"]) if fd_start <= r <= fd_end]
        im_idx = [i for i, r in enumerate(s1["residue_ids"]) if im_start <= r <= im_end]
        audit = audit_hamiltonian_sign(
            s1["sequence"], s1["coords"], fd_idx, im_idx
        )
        audit["gene"] = target.gene_name
        audits.append(audit)
        logger.info(
            f"  {target.gene_name}: ref==ground? {audit.get('ref_matches_ground')} "
            f"hamming={audit.get('hamming_ref_vs_ground')} "
            f"ref_agreement={audit.get('ref_agreement_with_ground', 0):.3f}"
        )

    n_match = sum(1 for a in audits if a.get("ref_matches_ground"))
    summary = {
        "n_audited": len(audits),
        "n_ref_matches_ground": n_match,
        "sign_flip_always_changes_ground": all(
            a.get("sign_flip_changes_ground", False) for a in audits
        ),
        "audits": audits,
        "bug_found": False,
        "bug_description": None,
    }

    # Check for systematic sign error: if ref NEVER matches ground but agreement is high,
    # that's model conflict not sign flip
    mean_hamming = float(
        np.mean([a.get("hamming_ref_vs_ground", 0) for a in audits])
    )
    summary["mean_hamming_ref_vs_ground"] = mean_hamming

    # Per-conformation energy vs TM on one protein
    data = load_protein_data(HELD_OUT_14[1])  # SRC
    if data:
        s1, s2 = data["s1"], data["s2"]
        cm = compute_contact_map(s1["coords"], threshold=8.0)
        built = build_ising_hamiltonian(
            s1["sequence"], cm, (data["fd_idx"], data["im_idx"])
        )
        H, n_q, selected = built
        model = pennylane_hamiltonian_to_ising_model(H, n_q)
        energies, tms = [], []
        from src.metrics.structural_metrics import tm_score

        for conf in data["ensemble"]:
            cm_c = compute_contact_map(conf["coords"], threshold=8.0)
            energies.append(conformation_ising_energy(cm_c, selected, model))
            valid = [i for i in data["ci1"] if i < len(conf["coords"])]
            tgt = data["ci2"][: len(valid)]
            tms.append(tm_score(s2["coords"][tgt], conf["coords"][valid]))
        rho_e, p_e = scipy_stats.spearmanr(energies, tms)
        summary["src_energy_vs_tm_spearman"] = {
            "rho": float(rho_e),
            "p": float(p_e),
        }
        logger.info(f"  SRC per-conf Ising energy vs TM→S2: rho={rho_e:.4f}, p={p_e:.4f}")

    logger.info(
        f"  Summary: {n_match}/{len(audits)} proteins have ref==ground bitstring, "
        f"mean Hamming={mean_hamming:.1f}"
    )
    return summary


def main():
    out = {}

    # Step 1
    step1 = run_step1_audit()
    out["step1"] = step1

    # Step 2
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Alanine dipeptide Ramachandran basin test")
    logger.info("=" * 70)
    step2 = alanine_dipeptide_basin_test()
    out["step2"] = step2
    if step2.get("basin_energies_full_hamiltonian"):
        for row in step2["basin_energies_full_hamiltonian"]:
            logger.info(
                f"  {row['basin']:8s} E={row['ising_energy']:+.4f} "
                f"agreement={row['agreement_with_ground']:.3f} contacts={row['n_contacts']}"
            )
        logger.info(
            f"  C7eq vs C7ax (full H): eq_mean={step2['eq_mean_energy']:.4f} "
            f"ax_mean={step2['ax_mean_energy']:.4f} "
            f"separates={step2['separates_basins_full_h']} p={step2['mannwhitney_eq_lower_p']:.4f}"
        )
        logger.info(
            f"  Independent-qubit ref==ground: {step2.get('independent_qubit_ref_matches_ground')}"
        )
        for row in step2.get("basin_energies_independent_qubit", []):
            logger.info(f"    ind {row['basin']:8s} E={row['ising_energy_independent']:+.4f}")

    # Baseline on 14 proteins: exact ground-state agreement (original QICESS quantum term)
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE: 14-protein ranking (ground-state agreement, original method)")
    logger.info("=" * 70)
    baseline = run_14_protein_ranking(method_exact_agreement, "exact_agreement")
    out["baseline_14"] = {
        k: v for k, v in baseline.items() if k != "per_protein"
    }
    logger.info(
        f"  mean Spearman rho = {baseline['mean_spearman_rho']:.4f} "
        f"(random {baseline['mean_random_spearman_rho']:.4f}, "
        f"Wilcoxon p={baseline['wilcoxon_p_vs_random']:.4f})"
    )

    # Step 3: one principled fix
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Principled fix — independent-qubit Ising, per-conformation energy")
    logger.info("=" * 70)
    fixed = run_14_protein_ranking(
        rank_by_independent_qubit_energy,
        "independent_qubit_per_conf_energy",
    )
    out["step3"] = {k: v for k, v in fixed.items() if k != "per_protein"}
    logger.info(
        f"  mean Spearman rho = {fixed['mean_spearman_rho']:.4f} "
        f"(random {fixed['mean_random_spearman_rho']:.4f}, "
        f"Wilcoxon p={fixed['wilcoxon_p_vs_random']:.4f})"
    )

    out_path = RESULTS_DIR / "discrimination_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"\nSaved to {out_path}")
    return out


if __name__ == "__main__":
    main()
