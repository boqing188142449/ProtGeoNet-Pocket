"""
Generates ESM-2 embeddings for protein sequences from PDB files.
"""

import sys
import os
import numpy as np
import torch
from collections import OrderedDict
import esm
from torch.cuda.amp import autocast


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device).eval()


AMINO_ACID_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def extract_sequence_from_pdb(pdb_path: str) -> str:
    """Extract amino acid sequence from a PDB file."""
    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    residues = OrderedDict()
    for line in lines:
        if line.startswith('ATOM'):
            try:
                residue_name = line[17:20].strip()
                residue_id = int(line[22:26].strip())
                if residue_name in AMINO_ACID_MAP:
                    residues[residue_id] = residue_name
            except ValueError:
                continue

    return ''.join(AMINO_ACID_MAP[residue] for residue in residues.values())


def get_esm2_embeddings(sequence: str, max_len: int = 1300) -> np.ndarray:
    """Generate ESM-2 embeddings for a protein sequence in chunks."""
    embeddings = []
    for i in range(0, len(sequence), max_len):
        chunk_sequence = sequence[i:i + max_len]
        data = [('protein', chunk_sequence)]
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        batch_tokens = batch_tokens.to(device)
        with autocast():
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        embeddings.append(results["representations"][33][0, 1:batch_lens[0] - 1].cpu().numpy())
        torch.cuda.empty_cache()

    return np.concatenate(embeddings, axis=0)


def process_protein(protein_folder: str) -> None:
    """Process protein subfolders to generate and save ESM-2 embeddings."""
    for folder_name in os.listdir(protein_folder):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        if not os.path.isdir(protein_subfolder):
            continue

        esm2_file = os.path.join(protein_subfolder, f"{folder_name}_esm.npy")
        if os.path.isfile(esm2_file):
            print(f"{esm2_file} already exists, skipping.")
            continue

        pdb_file = os.path.join(protein_subfolder, 'protein.pdb')
        if os.path.isfile(pdb_file):
            print(f"Processing protein: {folder_name}")
            sequence = extract_sequence_from_pdb(pdb_file)
            esm2_features = get_esm2_embeddings(sequence)
            np.save(esm2_file, esm2_features)
            print(f"Saved ESM-2 features to {esm2_file}")
        else:
            print(f"Error: protein.pdb not found in {protein_subfolder}")


def main() -> None:
    """Parse command-line arguments and process protein directory."""
    if len(sys.argv) != 2:
        print("Usage: python generate_esm_features_pdb.py <protein_folder>")
        sys.exit(1)

    protein_directory = sys.argv[1]
    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    process_protein(protein_directory)


if __name__ == "__main__":
    main()