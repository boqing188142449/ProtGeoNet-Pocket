"""
Generates ESM-2 embeddings for protein sequences from .mol2 files.
"""

import re
import sys
import os
import numpy as np
import torch
from collections import OrderedDict
import esm
from torch.cuda.amp import autocast


# Load ESM-2 model
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device).eval()


AMINO_ACID_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def extract_sequence_from_mol2(mol2_path: str) -> str:
    """Extract amino acid sequence from a .mol2 file."""
    with open(mol2_path, 'r') as file:
        lines = file.readlines()

    residues = OrderedDict()
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
                residue_id = int(match.group(7))
                residue_name = re.sub(r'\d+$', '', match.group(8))
                if residue_name in AMINO_ACID_MAP:
                    residues[residue_id] = residue_name

    return ''.join(AMINO_ACID_MAP[residue] for residue in residues.values())


def get_esm2_embeddings(sequence: str, max_len: int = 1024) -> np.ndarray:
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
        token_representations = results["representations"][33]
        embeddings.append(token_representations[0, 1:batch_lens[0] - 1].cpu().numpy())

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

        mol2_file = os.path.join(protein_subfolder, 'protein.mol2')
        if os.path.isfile(mol2_file):
            print(f"Processing protein: {folder_name}")
            sequence = extract_sequence_from_mol2(mol2_file)
            esm2_features = get_esm2_embeddings(sequence)
            np.save(esm2_file, esm2_features)
            print(f"Saved ESM-2 features to {esm2_file}")
        else:
            print(f"Error: protein.mol2 not found in {protein_subfolder}")


def main() -> None:
    """Parse command-line arguments and process protein directory."""
    if len(sys.argv) != 2:
        print("Usage: python generate_esm_features.py <protein_folder>")
        sys.exit(1)

    protein_directory = sys.argv[1]
    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    process_protein(protein_directory)


if __name__ == "__main__":
    main()