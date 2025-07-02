"""
Generates binding site labels for protein residues from PDB files based on ligand proximity.
"""

import numpy as np
import sys
import os
from collections import defaultdict
from scipy.spatial import KDTree


AMINO_ACID_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def load_pdb_ligand(pdb_path: str) -> np.ndarray:
    """Parse heavy atom coordinates from a ligand PDB file."""
    atom_coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                element = line[76:78].strip()
                if element != 'H':
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        atom_coords.append([x, y, z])
                    except ValueError:
                        continue

    if not atom_coords:
        raise ValueError(f"No heavy atom coordinates found in {pdb_path}")
    return np.array(atom_coords)


def load_pdb_protein(pdb_path: str) -> tuple:
    """Parse residue centroids, heavy atoms, IDs, and names from a protein PDB file."""
    residue_coords = defaultdict(list)
    residue_names = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                element = line[76:78].strip()
                if element != 'H':
                    try:
                        residue_name = line[17:20].strip()
                        residue_id = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        if residue_name in AMINO_ACID_MAP:
                            residue_coords[residue_id].append([x, y, z])
                            residue_names[residue_id] = residue_name
                    except ValueError:
                        continue

    residue_centers = []
    residue_atoms = []
    residue_ids = []
    residue_name_list = []
    for res_id, coords in residue_coords.items():
        if coords:
            coords_array = np.array(coords)
            residue_centers.append(np.mean(coords_array, axis=0))
            residue_atoms.append(coords_array)
            residue_ids.append(res_id)
            residue_name_list.append(residue_names[res_id])

    return np.array(residue_centers), residue_atoms, residue_ids, residue_name_list


def label_binding_sites(protein_pdb: str, ligand_pdb: str, cutoff: float = 10.0) -> tuple:
    """Label residues as binding sites if within cutoff distance of ligand heavy atoms."""
    ligand_heavy_atoms = load_pdb_ligand(ligand_pdb)
    residue_centers, residue_atoms, residue_ids, residue_names = load_pdb_protein(protein_pdb)
    labels = np.zeros(len(residue_ids))
    ligand_tree = KDTree(ligand_heavy_atoms)

    for i, res_atoms in enumerate(residue_atoms):
        min_distance = np.min(ligand_tree.query(res_atoms)[0])
        labels[i] = 1 if min_distance <= cutoff else 0

    return labels, residue_ids, residue_names, ligand_heavy_atoms


def process_protein(protein_folder: str) -> None:
    """Process protein subfolders to generate and save binding site labels."""
    for folder_name in os.listdir(protein_folder):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        if not os.path.isdir(protein_subfolder):
            continue

        protein_pdb = os.path.join(protein_subfolder, 'protein.pdb')
        ligand_pdb = os.path.join(protein_subfolder, 'ligand.pdb')

        if os.path.isfile(protein_pdb) and os.path.isfile(ligand_pdb):
            print(f"Processing protein: {folder_name}")
            labels, _, _, _ = label_binding_sites(protein_pdb, ligand_pdb)
            labels_path = os.path.join(protein_subfolder, f"{folder_name}_labels.npy")
            np.save(labels_path, labels)
            print(f"Saved labels to {labels_path}")
        else:
            print(f"Error: protein.pdb or ligand.pdb not found in {protein_subfolder}")


def main() -> None:
    """Parse command-line arguments and process protein directory."""
    if len(sys.argv) != 2:
        print("Usage: python generate_binding_labels_pdb.py <protein_folder>")
        sys.exit(1)

    protein_directory = sys.argv[1]
    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    process_protein(protein_directory)


if __name__ == "__main__":
    main()