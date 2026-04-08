"""
pdb_fetcher.py — Download and parse real PDB structures from RCSB.
No synthetic data. Every structure is experimentally determined.
"""
import os
import requests
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import StringIO

logger = logging.getLogger(__name__)

PDB_DIR = Path(__file__).parent.parent.parent / "data" / "pdb_cache"
PDB_DIR.mkdir(parents=True, exist_ok=True)

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
AFDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

# Standard amino acid 3-letter to 1-letter mapping
AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

AA1_TO_IDX = {a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}


def fetch_pdb(pdb_id: str, force: bool = False) -> Optional[str]:
    """Download PDB file from RCSB. Returns path to local file."""
    pdb_id = pdb_id.upper().strip()
    local_path = PDB_DIR / f"{pdb_id}.pdb"
    
    if local_path.exists() and not force:
        return str(local_path)
    
    url = RCSB_URL.format(pdb_id=pdb_id)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        local_path.write_text(resp.text)
        logger.info(f"Downloaded {pdb_id} from RCSB ({len(resp.text)} bytes)")
        return str(local_path)
    except Exception as e:
        logger.error(f"Failed to fetch {pdb_id}: {e}")
        return None


def fetch_alphafold_prediction(uniprot_id: str, force: bool = False) -> Optional[str]:
    """Download AlphaFold DB prediction (AF2 v4). Returns path."""
    local_path = PDB_DIR / f"AF-{uniprot_id}-F1.pdb"
    
    if local_path.exists() and not force:
        return str(local_path)
    
    url = AFDB_URL.format(uniprot_id=uniprot_id)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        local_path.write_text(resp.text)
        logger.info(f"Downloaded AF prediction for {uniprot_id}")
        return str(local_path)
    except Exception as e:
        logger.error(f"Failed to fetch AF prediction for {uniprot_id}: {e}")
        return None


def parse_pdb_ca_coords(pdb_path: str, chain: str = None, 
                         res_range: Tuple[int, int] = None) -> Dict:
    """
    Parse Cα coordinates from a PDB file.
    
    Returns:
        dict with keys:
            'coords': np.array of shape (N, 3) — Cα xyz coordinates in Å
            'sequence': str — one-letter amino acid sequence
            'residue_ids': list of int — residue numbers
            'chain': str — chain ID used
            'bfactors': np.array — B-factors (proxy for flexibility/confidence)
    """
    coords = []
    sequence = []
    residue_ids = []
    bfactors = []
    chain_id = chain
    
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            
            res_name = line[17:20].strip()
            if res_name not in AA3TO1:
                continue
            
            this_chain = line[21].strip()
            if chain_id is None:
                chain_id = this_chain  # Use first chain
            elif this_chain != chain_id:
                continue
            
            res_num = int(line[22:26].strip())
            
            if res_range is not None:
                if res_num < res_range[0] or res_num > res_range[1]:
                    continue
            
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            bf = float(line[60:66].strip()) if len(line) >= 66 else 0.0
            
            coords.append([x, y, z])
            sequence.append(AA3TO1[res_name])
            residue_ids.append(res_num)
            bfactors.append(bf)
    
    if not coords:
        logger.warning(f"No Cα atoms found in {pdb_path} (chain={chain})")
        return None
    
    return {
        'coords': np.array(coords, dtype=np.float64),
        'sequence': ''.join(sequence),
        'residue_ids': residue_ids,
        'chain': chain_id,
        'bfactors': np.array(bfactors),
        'pdb_path': pdb_path,
        'n_residues': len(coords)
    }


def parse_pdb_all_atom(pdb_path: str, chain: str = None) -> Dict:
    """Parse all heavy-atom coordinates from PDB, organized by residue."""
    residues = {}
    chain_id = chain
    
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            if res_name not in AA3TO1:
                continue
            
            this_chain = line[21].strip()
            if chain_id is None:
                chain_id = this_chain
            elif this_chain != chain_id:
                continue
            
            # Skip hydrogens
            element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
            if element == 'H':
                continue
            
            res_num = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            
            if res_num not in residues:
                residues[res_num] = {
                    'name': res_name,
                    'aa': AA3TO1[res_name],
                    'atoms': {}
                }
            residues[res_num]['atoms'][atom_name] = np.array([x, y, z])
    
    return {'residues': residues, 'chain': chain_id}


def compute_contact_map(coords: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """
    Compute binary contact map from Cα coordinates.
    Contact defined as Cα-Cα distance < threshold (default 8 Å).
    """
    n = len(coords)
    dists = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
    contacts = (dists < threshold).astype(np.float64)
    # Zero out trivial contacts (i, i) and near-neighbors (|i-j| <= 2)
    for i in range(n):
        for j in range(max(0, i - 2), min(n, i + 3)):
            contacts[i, j] = 0.0
    return contacts


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Cα distance matrix."""
    return np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))


def compute_phi_psi(pdb_path: str, chain: str = None) -> List[Tuple[float, float]]:
    """
    Compute backbone dihedral angles (φ, ψ) from PDB.
    Returns list of (phi, psi) tuples in degrees.
    """
    # Parse backbone atoms: N, CA, C for each residue
    backbone = {}  # res_num -> {'N': xyz, 'CA': xyz, 'C': xyz}
    chain_id = chain
    
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in ('N', 'CA', 'C'):
                continue
            res_name = line[17:20].strip()
            if res_name not in AA3TO1:
                continue
            this_chain = line[21].strip()
            if chain_id is None:
                chain_id = this_chain
            elif this_chain != chain_id:
                continue
            res_num = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            if res_num not in backbone:
                backbone[res_num] = {}
            backbone[res_num][atom_name] = np.array([x, y, z])
    
    sorted_res = sorted(backbone.keys())
    angles = []
    
    for idx in range(len(sorted_res)):
        res = sorted_res[idx]
        phi, psi = np.nan, np.nan
        
        # φ(i) = dihedral(C(i-1), N(i), CA(i), C(i))
        if idx > 0:
            prev = sorted_res[idx - 1]
            if all(a in backbone.get(prev, {}) for a in ['C']) and \
               all(a in backbone.get(res, {}) for a in ['N', 'CA', 'C']):
                phi = _dihedral(backbone[prev]['C'], backbone[res]['N'],
                               backbone[res]['CA'], backbone[res]['C'])
        
        # ψ(i) = dihedral(N(i), CA(i), C(i), N(i+1))
        if idx < len(sorted_res) - 1:
            next_res = sorted_res[idx + 1]
            if all(a in backbone.get(res, {}) for a in ['N', 'CA', 'C']) and \
               'N' in backbone.get(next_res, {}):
                psi = _dihedral(backbone[res]['N'], backbone[res]['CA'],
                               backbone[res]['C'], backbone[next_res]['N'])
        
        angles.append((phi, psi))
    
    return angles


def _dihedral(p0, p1, p2, p3) -> float:
    """Compute dihedral angle in degrees between four points."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0
    
    n1 /= n1_norm
    n2 /= n2_norm
    
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    return np.degrees(np.arctan2(y, x))
