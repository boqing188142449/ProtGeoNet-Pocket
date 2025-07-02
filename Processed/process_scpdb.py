
"""
Copies protein and ligand .mol2 files from scPDB database folders to a new directory structure.
"""

import os
import sys
import shutil


def copy_mol2_files(protein_folder: str, output_base_path: str) -> None:
    """Copy protein and ligand .mol2 files to a new folder named after the protein."""
    folder_name = os.path.basename(protein_folder)
    new_folder = os.path.join(output_base_path, folder_name)
    os.makedirs(new_folder, exist_ok=True)

    protein_file = ligand_file = None
    for file_name in os.listdir(protein_folder):
        if file_name.lower().endswith('protein.mol2'):
            protein_file = os.path.join(protein_folder, file_name)
        elif file_name.lower().endswith('ligand.mol2'):
            ligand_file = os.path.join(protein_folder, file_name)

    if protein_file:
        new_protein_path = os.path.join(new_folder, 'protein.mol2')
        shutil.copy(protein_file, new_protein_path)
        print(f"Copied protein file to: {new_protein_path}")

    if ligand_file:
        new_ligand_path = os.path.join(new_folder, 'ligand.mol2')
        shutil.copy(ligand_file, new_ligand_path)
        print(f"Copied ligand file to: {new_ligand_path}")
    else:
        try:
            os.rmdir(new_folder)
            print(f"No ligand found, removed empty folder: {new_folder}")
        except OSError:
            print(f"Could not remove non-empty folder: {new_folder}")


def process_protein(protein_directory: str, output_base_path: str) -> None:
    """Process all subfolders in the protein directory."""
    protein_folders = [
        os.path.join(protein_directory, folder_name)
        for folder_name in os.listdir(protein_directory)
        if os.path.isdir(os.path.join(protein_directory, folder_name))
    ]
    for protein_folder in protein_folders:
        copy_mol2_files(protein_folder, output_base_path)


def main() -> None:
    """Parse command-line arguments and process protein directory."""
    if len(sys.argv) != 3:
        print("Usage: python process_scpdb.py <protein_folder> <output_folder>")
        sys.exit(1)

    protein_directory, output_base_path = sys.argv[1], sys.argv[2]

    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    os.makedirs(output_base_path, exist_ok=True)
    process_protein(protein_directory, output_base_path)


if __name__ == "__main__":
    main()