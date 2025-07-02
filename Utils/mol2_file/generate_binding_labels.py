"""
Generates binding site labels for protein residues based on ligand proximity.
"""

import numpy as np
import re
import sys
import os
from collections import defaultdict


AMINO_ACID_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def parse_ligand_mol2(mol2_path: str) -> np.ndarray:
    """Parse heavy atom coordinates from a ligand .mol2 file."""
    atom_coords = []
    with open(mol2_path, 'r') as f:
        lines = f.readlines()

    atom_section = False
    for line in lines:
        if line.startswith('@<TRIPOS>ATOM'):
            atom_section = True
            continue
        elif line.startswith('@<TRIPOS>'):
            atom_section = False
        if atom_section and len(line.split()) >= 6:
            try:
                x, y, z = map(float, line.split()[2:5])
                atom_type = line.split()[5]
                if not atom_type.startswith('H'):
                    atom_coords.append([x, y, z])
            except ValueError:
                continue

    if not atom_coords:
        raise ValueError(f"No heavy atom coordinates found in {mol2_path}")
    return np.array(atom_coords)


def load_protein_mol2(mol2_path: str) -> tuple:
    """Parse residue centroids, IDs, names, and atom coordinates from a protein .mol2 file."""
    residue_coords = defaultdict(list)
    residue_names = {}
    with open(mol2_path, 'r') as file:
        lines = file.readlines()

    in_atom_section = False
    for line in lines:
        if line.strip() == "@<TRIPOS>ATOM":
            in_atom_section = True
            continue
        if in_atom_section and line.strip() == "@<TRIPOS>BOND":
            break
        if in_atom_section:
            match = re.match(
                r'\s*(\d+)\s+([A-Za-z0-9]+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([A-Za-z0-9.]+)\s+(\d+)\s+([A-Za-z0-9]+)\s+([0-9.-]+)',
                line)
            if match:
                _, _, x, y, z, _, residue_id, residue_name, _ = match.groups()
                residue_name = re.sub(r'\d+$', '', residue_name)
                if residue_name in AMINO_ACID_MAP:
                    residue_coords[int(residue_id)].append([float(x), float(y), float(z)])
                    residue_names[int(residue_id)] = residue_name

    residue_centers = []
    residue_ids = []
    residue_name_list = []
    residue_atom_coords = []
    for res_id, coords in residue_coords.items():
        if coords:
            coords_array = np.array(coords)
            residue_centers.append(np.mean(coords_array, axis=0))
            residue_ids.append(res_id)
            residue_name_list.append(residue_names[res_id])
            residue_atom_coords.append(coords_array)

    return np.array(residue_centers), residue_ids, residue_name_list, residue_atom_coords


def label_binding_sites(protein_mol2: str, ligand_mol2: str, cutoff: float = 10.0) -> tuple:
    """Label residues as binding sites if within cutoff distance of ligand heavy atoms."""
    ligand_heavy_atoms = parse_ligand_mol2(ligand_mol2)
    residue_centers, residue_ids, residue_names, residue_atoms_list = load_protein_mol2(protein_mol2)

    labels = np.zeros(len(residue_ids))
    for i, atoms in enumerate(residue_atoms_list):
        dists = np.linalg.norm(atoms[:, None, :] - ligand_heavy_atoms[None, :, :], axis=2)
        if np.any(dists <= cutoff):
            labels[i] = 1

    return labels, residue_ids, residue_names, ligand_heavy_atoms


def process_protein(protein_folder: str) -> None:
    """Process protein subfolders to generate and save binding site labels."""
    for folder_name in os.listdir(protein_folder):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        if not os.path.isdir(protein_subfolder):
            continue

        protein_mol2 = os.path.join(protein_subfolder, 'protein.mol2')
        ligand_mol2 = os.path.join(protein_subfolder, 'ligand.mol2')

        if os.path.isfile(protein_mol2) and os.path.isfile(ligand_mol2):
            print(f"Processing protein: {folder_name}")
            labels, _, _, _ = label_binding_sites(protein_mol2, ligand_mol2)
            label_save_path = os.path.join(protein_subfolder, f"{folder_name}_labels.npy")
            np.save(label_save_path, labels)
            print(f"Saved labels to {label_save_path}")
        else:
            print(f"Error: protein.mol2 or ligand.mol2 not found in {protein_subfolder}")


def main() -> None:
    """Parse command-line arguments and process protein directory."""
    if len(sys.argv) != 2:
        print("Usage: python generate_binding_labels.py <protein_folder>")
        sys.exit(1)

    protein_directory = sys.argv[1]
    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    process_protein(protein_directory)


if __name__ == "__main__":
    main()