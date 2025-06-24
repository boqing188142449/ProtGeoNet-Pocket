"""
Extracts residue centroids and ligand heavy atom coordinates from .mol2 files.
"""

import re
import sys
import os
import numpy as np
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


def load_protein_mol2(mol2_path: str) -> np.ndarray:
    """Parse residue centroids from a protein .mol2 file."""
    residue_coords = defaultdict(list)
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
                residue_name = re.sub(r'\d+$', '', match.group(8))
                if residue_name in AMINO_ACID_MAP:
                    residue_coords[int(match.group(7))].append([float(match.group(3)), float(match.group(4)), float(match.group(5))])

    centers = [np.mean(np.array(coords), axis=0) for coords in residue_coords.values()]
    return np.array(centers)


def process_protein(protein_folder: str) -> None:
    """Process protein subfolders to generate and save coordinates."""
    for folder_name in os.listdir(protein_folder):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        if not os.path.isdir(protein_subfolder):
            continue

        protein_mol2 = os.path.join(protein_subfolder, 'protein.mol2')
        ligand_mol2 = os.path.join(protein_subfolder, 'ligand.mol2')

        if os.path.isfile(protein_mol2) and os.path.isfile(ligand_mol2):
            print(f"Processing protein: {folder_name}")
            protein_coords = load_protein_mol2(protein_mol2)
            ligand_coords = parse_ligand_mol2(ligand_mol2)

            np.save(os.path.join(protein_subfolder, f"{folder_name}_protein_coords.npy"), protein_coords)
            np.save(os.path.join(protein_subfolder, f"{folder_name}_ligand_coords.npy"), ligand_coords)
            print(f"Saved coordinates to {protein_subfolder}")
        else:
            print(f"Error: protein.mol2 or ligand.mol2 not found in {protein_subfolder}")


def main() -> None:
    """Parse command-line arguments and process protein directory."""
    if len(sys.argv) != 2:
        print("Usage: python generate_coordinates.py <protein_folder>")
        sys.exit(1)

    protein_directory = sys.argv[1]
    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    process_protein(protein_directory)


if __name__ == "__main__":
    main()