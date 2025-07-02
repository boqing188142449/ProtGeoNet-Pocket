"""
Generates ProtT5 embeddings for protein sequences from .mol2 files.
"""

import re
import sys
import os
import numpy as np
import torch
from collections import OrderedDict
from transformers import T5Tokenizer, T5EncoderModel


# Load ProtT5 model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cache_dir = os.getenv('MODEL_CACHE_DIR', '/media/2t/zhangzhi/ProtGeoNet-Pocket/model_cache')
tokenizer = T5Tokenizer.from_pretrained(cache_dir, do_lower_case=False)
model = T5EncoderModel.from_pretrained(cache_dir).to(device)
if device == torch.device("cpu"):
    model = model.to(torch.float32)


AMINO_ACID_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def generate_embeddings(sequence: str) -> np.ndarray:
    """Generate ProtT5 embeddings for a protein sequence."""
    processed_sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    ids = tokenizer([processed_sequence], add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    return embedding_repr.last_hidden_state[0, :len(sequence)].cpu().numpy()


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


def process_protein(protein_folder: str) -> None:
    """Process protein subfolders to generate and save ProtT5 embeddings."""
    for folder_name in os.listdir(protein_folder):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        if not os.path.isdir(protein_subfolder):
            continue

        prott5_file = os.path.join(protein_subfolder, f"{folder_name}_prott5.npy")
        if os.path.isfile(prott5_file):
            print(f"{prott5_file} already exists, skipping.")
            continue

        mol2_file = os.path.join(protein_subfolder, 'protein.mol2')
        if os.path.isfile(mol2_file):
            print(f"Processing protein: {folder_name}")
            sequence = extract_sequence_from_mol2(mol2_file)
            prott5_features = generate_embeddings(sequence)
            np.save(prott5_file, prott5_features)
            print(f"Saved ProtT5 features to {prott5_file}")
        else:
            print(f"Error: protein.mol2 not found in {protein_subfolder}")


def main() -> None:
    """Parse command-line arguments and process protein directory."""
    if len(sys.argv) != 2:
        print("Usage: python generate_prott5_features.py <protein_folder>")
        sys.exit(1)

    protein_directory = sys.argv[1]
    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    process_protein(protein_directory)


if __name__ == "__main__":
    main()