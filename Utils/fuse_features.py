"""
Combines ESM-2 and ProtT5 features into a single feature file for each protein.
"""

import sys
import os
import numpy as np


def process_protein(protein_folder: str) -> None:
    """
    Process protein subfolders to combine ESM-2 and ProtT5 features.

    Args:
        protein_folder (str): Directory containing protein subfolders.
    """
    for folder_name in os.listdir(protein_folder):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        if not os.path.isdir(protein_subfolder):
            continue

        print(f"Processing protein: {folder_name}")
        esm_file = os.path.join(protein_subfolder, f"{folder_name}_esm.npy")
        prott5_file = os.path.join(protein_subfolder, f"{folder_name}_prott5.npy")
        output_file = os.path.join(protein_subfolder, f"{folder_name}_features.npy")

        if os.path.isfile(output_file):
            print(f"{output_file} already exists, skipping.")
            continue

        if os.path.isfile(esm_file) and os.path.isfile(prott5_file):
            try:
                esm_features = np.load(esm_file)
                prott5_features = np.load(prott5_file)
                combined_features = np.concatenate((esm_features, prott5_features), axis=1)
                np.save(output_file, combined_features)
                print(f"Saved combined features to {output_file} (shape: {combined_features.shape})")
            except Exception as e:
                print(f"Error combining features for protein {folder_name}: {e}")
        else:
            missing_files = [f for f in [esm_file, prott5_file] if not os.path.isfile(f)]
            print(f"Error: Missing files for protein {folder_name}: {missing_files}")


def main() -> None:
    """Parse command-line arguments and process protein directory."""
    if len(sys.argv) != 2:
        print("Usage: python combine_protein_features.py <protein_folder>")
        sys.exit(1)

    protein_directory = sys.argv[1]
    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    process_protein(protein_directory)


if __name__ == "__main__":
    main()