"""
Copies and renames protein and ligand .mol2 files from SC6K dataset folders.
"""

import os
import sys
import shutil


def copy_and_rename_files(protein_folder: str, output_base_path: str) -> None:
    """Copy and rename ligand and protein .mol2 files to a new folder."""
    for file_name in os.listdir(protein_folder):
        if file_name.lower().count('_') == 2 and 'PROT' not in file_name.upper():
            ligand_name = os.path.splitext(file_name)[0]
            new_folder = os.path.join(output_base_path, ligand_name)
            os.makedirs(new_folder, exist_ok=True)

            new_ligand_path = os.path.join(new_folder, 'ligand.mol2')
            shutil.copy(os.path.join(protein_folder, file_name), new_ligand_path)
            print(f"Copied and renamed ligand file to: {new_ligand_path}")

            protein_file_name = f"{ligand_name}_PROT.mol2"
            protein_file_path = os.path.join(protein_folder, protein_file_name)
            if os.path.isfile(protein_file_path):
                new_protein_path = os.path.join(new_folder, 'protein.mol2')
                shutil.copy(protein_file_path, new_protein_path)
                print(f"Copied and renamed protein file to: {new_protein_path}")
            else:
                try:
                    os.rmdir(new_folder)
                    print(f"No protein file found, removed empty folder: {new_folder}")
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
        copy_and_rename_files(protein_folder, output_base_path)


def main() -> None:
    """Parse command-line arguments and process protein directory."""
    if len(sys.argv) != 3:
        print("Usage: python process_sc6k.py <protein_folder> <output_folder>")
        sys.exit(1)

    protein_directory, output_base_path = sys.argv[1], sys.argv[2]

    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    os.makedirs(output_base_path, exist_ok=True)
    process_protein(protein_directory, output_base_path)


if __name__ == "__main__":
    main()