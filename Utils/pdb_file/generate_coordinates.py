"""
Extracts residue centroids and ligand heavy atom coordinates from PDB files.
"""

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


def load_pdb_protein(pdb_path: str) -> np.ndarray:
    """Parse residue centroids from a protein PDB file."""
    residue_coords = defaultdict(list)
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                try:
                    residue_name = line[17:20].strip()
                    residue_id = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    if residue_name in AMINO_ACID_MAP:
                        residue_coords[residue_id].append([x, y, z])
                except ValueError:
                    continue

    centers = [np.mean(np.array(coords), axis=0) for coords in residue_coords.values()]
    return np.array(centers)


def process_protein(protein_folder: str) -> None:
    """Process protein subfolders to generate and save coordinates."""
    for folder_name in os.listdir(protein_folder):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        if not os.path.isdir(protein_subfolder):
            continue

        protein_pdb = os.path.join(protein_subfolder, 'protein.pdb')
        ligand_pdb = os.path.join(protein_subfolder, 'ligand.pdb')

        if os.path.isfile(protein_pdb) and os.path.isfile(ligand_pdb):
            print(f"Processing protein: {folder_name}")
            protein_coords = load_pdb_protein(protein_pdb)
            ligand_coords = load_pdb_ligand(ligand_pdb)

            np.save(os.path.join(protein_subfolder, f"{folder_name}_protein_coords.npy"), protein_coords)
            np.save(os.path.join(protein_subfolder, f"{folder_name}_ligand_coords.npy"), ligand_coords)
            print(f"Saved coordinates to {protein_subfolder}")
        else:
            print(f"Error: protein.pdb or ligand.pdb not found in {protein_subfolder}")


def main() -> None:
    """Parse command-line arguments and process protein directory."""
    if len(sys.argv) != 2:
        print("Usage: python generate_coordinates_pdb.py <protein_folder>")
        sys.exit(1)

    protein_directory = sys.argv[1]
    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    process_protein(protein_directory)


if __name__ == "__main__":
    main()