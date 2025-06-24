"""
Creates a PyTorch Geometric dataset for protein graphs from preprocessed .npy files.
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, Batch


class ProteinGraphDataset(Dataset):
    """Dataset for protein graphs with features, edges, labels, and coordinates."""
    def __init__(self, root: str):
        """
        Initialize the dataset.

        Args:
            root (str): Root directory containing protein subfolders.
        """
        self.root = root
        super().__init__(root)

    def len(self) -> int:
        """Return the number of proteins in the dataset."""
        return len(os.listdir(self.root))

    def get(self, idx: int) -> Data:
        """
        Load data for a single protein.

        Args:
            idx (int): Index of the protein.

        Returns:
            Data: PyTorch Geometric Data object with features, edges, labels, and positions.
        """
        protein_folder = os.listdir(self.root)[idx]
        folder_path = os.path.join(self.root, protein_folder)

        # Construct file paths
        features_path = os.path.join(folder_path, f"{protein_folder}_features.npy")
        edge_index_path = os.path.join(folder_path, f"{protein_folder}_edge_index.npy")
        edge_attr_path = os.path.join(folder_path, f"{protein_folder}_edge_attr.npy")
        label_path = os.path.join(folder_path, f"{protein_folder}_labels.npy")
        protein_pos_path = os.path.join(folder_path, f"{protein_folder}_protein_coords.npy")
        ligand_pos_path = os.path.join(folder_path, f"{protein_folder}_ligand_coords.npy")

        # Load data
        try:
            features = np.load(features_path)
            edge_index = np.load(edge_index_path)
            edge_attr = np.load(edge_attr_path)
            labels = np.load(label_path)
            protein_pos = np.load(protein_pos_path)
            ligand_pos = np.load(ligand_pos_path) if os.path.exists(ligand_pos_path) else np.zeros((1, 3))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing file for protein {protein_folder}: {e.filename}")

        # Convert to tensors
        x = torch.from_numpy(features).float()
        edge_index = torch.from_numpy(edge_index).long()
        edge_attr = torch.from_numpy(edge_attr).float()
        y = torch.from_numpy(labels).long()
        protein_pos = torch.from_numpy(protein_pos).float()
        ligand_pos = torch.from_numpy(ligand_pos).float()

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            protein_pos=protein_pos,
            ligand_pos=ligand_pos
        )

    def get_all(self) -> list:
        """
        Load data for all proteins in the dataset.

        Returns:
            list: List of PyTorch Geometric Data objects.
        """
        data_list = []
        for protein_folder in os.listdir(self.root):
            folder_path = os.path.join(self.root, protein_folder)

            # Construct file paths
            features_path = os.path.join(folder_path, f"{protein_folder}_features.npy")
            edge_index_path = os.path.join(folder_path, f"{protein_folder}_edge_index.npy")
            edge_attr_path = os.path.join(folder_path, f"{protein_folder}_edge_attr.npy")
            label_path = os.path.join(folder_path, f"{protein_folder}_labels.npy")
            protein_pos_path = os.path.join(folder_path, f"{protein_folder}_protein_coords.npy")
            ligand_pos_path = os.path.join(folder_path, f"{protein_folder}_ligand_coords.npy")

            # Load data
            try:
                features = np.load(features_path)
                edge_index = np.load(edge_index_path)
                edge_attr = np.load(edge_attr_path)
                labels = np.load(label_path)
                protein_pos = np.load(protein_pos_path)
                ligand_pos = np.load(ligand_pos_path) if os.path.exists(ligand_pos_path) else np.zeros((1, 3))
            except FileNotFoundError as e:
                print(f"Skipping protein {protein_folder}: Missing file {e.filename}")
                continue

            # Convert to tensors
            x = torch.from_numpy(features).float()
            edge_index = torch.from_numpy(edge_index).long()
            edge_attr = torch.from_numpy(edge_attr).float()
            y = torch.from_numpy(labels).long()
            protein_pos = torch.from_numpy(protein_pos).float()
            ligand_pos = torch.from_numpy(ligand_pos).float()

            data_list.append(Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                protein_pos=protein_pos,
                ligand_pos=ligand_pos
            ))

        return data_list

    @staticmethod
    def collate_fn(batch: list) -> Batch:
        """
        Collate a list of Data objects into a single Batch.

        Args:
            batch (list): List of Data objects.

        Returns:
            Batch: PyTorch Geometric Batch object.
        """
        return Batch.from_data_list(batch)

