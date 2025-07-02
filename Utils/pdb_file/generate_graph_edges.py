"""
Generates graph edges and features for protein residues from PDB files.
"""

import os
import sys
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist


AMINO_ACID_MAP = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}


def compute_centers_and_types_from_pdb(pdb_path: str) -> tuple:
    """Compute residue centroids and types from a PDB file."""
    residue_coords = defaultdict(list)
    residue_types = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    residue_name = line[17:20].strip()
                    residue_id = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    if residue_name in AMINO_ACID_MAP:
                        residue_coords[residue_id].append([x, y, z])
                        residue_types[residue_id] = residue_name
                except (ValueError, IndexError):
                    continue

    sorted_ids = sorted(residue_coords.keys())
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
    centers = [np.mean(np.array(residue_coords[rid]), axis=0) for rid in sorted_ids]
    types = [AMINO_ACID_MAP[residue_types[rid]] for rid in sorted_ids]

    return np.array(centers), np.array(types), id_map, len(sorted_ids)


def compute_adjacency_matrix(centers: np.ndarray, radius: float = 4.0) -> np.ndarray:
    """Compute adjacency matrix based on residue centroid distances."""
    adj_matrix = (cdist(centers, centers) < radius).astype(np.float32)
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix


def adjacency_to_edges_and_features(adj_matrix: np.ndarray, centers: np.ndarray, types: np.ndarray) -> tuple:
    """Convert adjacency matrix to edge indices and features."""
    row, col = np.nonzero(adj_matrix)
    edge_index = np.vstack((row, col))

    centers_rbf = np.linspace(0, 10, 16)
    sigma = 0.5
    edge_attr = []
    for i, j in zip(row, col):
        dist = np.linalg.norm(centers[j] - centers[i])
        inv_dist = 1.0 / (dist + 1e-8)
        direction = (centers[j] - centers[i]) / (dist + 1e-8)
        rbf = np.exp(-((dist - centers_rbf) ** 2) / (2 * sigma ** 2))
        edge_attr.append(np.concatenate(([inv_dist], direction, rbf)))

    return edge_index, np.array(edge_attr)


def process_protein(protein_folder: str) -> None:
    """Process protein subfolders to generate and save graph edges."""
    for folder_name in os.listdir(protein_folder):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        if not os.path.isdir(protein_subfolder):
            continue

        pdb_file = os.path.join(protein_subfolder, 'protein.pdb')
        if not os.path.isfile(pdb_file):
            print(f"Missing protein.pdb in {protein_subfolder}")
            continue

        print(f"Processing protein: {folder_name}")
        centers, types, _, residue_count = compute_centers_and_types_from_pdb(pdb_file)
        adj_matrix = compute_adjacency_matrix(centers)
        edge_index, edge_attr = adjacency_to_edges_and_features(adj_matrix, centers, types)

        np.save(os.path.join(protein_subfolder, f"{folder_name}_edge_index.npy"), edge_index)
        np.save(os.path.join(protein_subfolder, f"{folder_name}_edge_attr.npy"), edge_attr)
        print(f"Saved edge data for {folder_name}: {adj_matrix.shape}, {edge_index.shape}, {edge_attr.shape}")


def main() -> None:
    """Parse command-line arguments and process protein directory."""
    if len(sys.argv) != 2:
        print("Usage: python generate_graph_edges_pdb.py <protein_folder>")
        sys.exit(1)

    protein_directory = sys.argv[1]
    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    process_protein(protein_directory)


if __name__ == "__main__":
    main()