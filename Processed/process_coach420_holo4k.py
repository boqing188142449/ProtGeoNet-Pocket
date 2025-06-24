"""
Extracts protein and ligand atoms from PDB files in Coach420 and Holo4k datasets.
"""

import os
import sys


def extract_protein_and_ligand(pdb_file_path: str, output_base_path: str) -> None:
    """Extract protein and ligand atoms from a PDB file and save to separate files."""
    protein_name = os.path.basename(pdb_file_path).split('.')[0]
    protein_folder = os.path.join(output_base_path, protein_name)
    os.makedirs(protein_folder, exist_ok=True)

    with open(pdb_file_path, 'r') as pdb_file:
        lines = pdb_file.readlines()

    protein_atoms = [line for line in lines if line.startswith('ATOM')]
    ligand_atoms = [
        line for line in lines
        if line.startswith('HETATM') and line[17:20].strip() != 'HOH'
    ]

    protein_output_path = os.path.join(protein_folder, 'protein.pdb')
    with open(protein_output_path, 'w') as protein_file:
        protein_file.writelines(protein_atoms)

    if ligand_atoms:
        ligand_output_path = os.path.join(protein_folder, 'ligand.pdb')
        with open(ligand_output_path, 'w') as ligand_file:
            ligand_file.writelines(ligand_atoms)
        print(f"Saved protein and ligand to: {protein_output_path}, {ligand_output_path}")
    else:
        try:
            os.rmdir(protein_folder)
            print(f"No ligand detected, removed empty folder: {protein_folder}")
        except OSError:
            pass


def process_protein(protein_directory: str, output_base_path: str) -> None:
    """Process all PDB files in the protein directory."""
    pdb_files = [f for f in os.listdir(protein_directory) if f.lower().endswith('.pdb')]
    for pdb_file in pdb_files:
        pdb_file_path = os.path.join(protein_directory, pdb_file)
        extract_protein_and_ligand(pdb_file_path, output_base_path)


def main() -> None:
    """Parse command-line arguments and process protein directory."""
    if len(sys.argv) != 3:
        print("Usage: python process_coach420_holo4k.py <protein_folder> <output_folder>")
        sys.exit(1)

    protein_directory, output_base_path = sys.argv[1], sys.argv[2]

    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    os.makedirs(output_base_path, exist_ok=True)
    process_protein(protein_directory, output_base_path)


if __name__ == "__main__":
    main()