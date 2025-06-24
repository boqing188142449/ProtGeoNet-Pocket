"""
Generates ProtT5 embeddings for protein sequences from PDB files.
"""

import sys
import os
import numpy as np
import torch
from collections import OrderedDict
from transformers import T5Tokenizer, T5EncoderModel


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


def process_protein(protein_folder: str) -> None:
    """Process protein subfolders to generate and save ProtT5 embeddings."""
    for folder_name in os.listdir(protein_folder):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        if not os.path.isdir(protein_subfolder):
            continue

        portt5_file = os.path.join(protein_subfolder, f"{folder_name}_portt5.npy")
        if os.path.isfile(portt5_file):
            print(f"{portt5_file} already exists, skipping.")
            continue

        pdb_file = os.path.join(protein_subfolder, 'protein.pdb')
        if os.path.isfile(pdb_file):
            print(f"Processing protein: {folder_name}")
            sequence = extract_sequence_from_pdb(pdb_file)
            portt5_features = generate_embeddings(sequence)
            np.save(portt5_file, portt5_features)
            print(f"Saved ProtT5 features to {portt5_file}")
        else:
            print(f"Error: protein.pdb not found in {protein_subfolder}")


def main() -> None:
    """Parse command-line arguments and process protein directory."""
    if len(sys.argv) != 2:
        print("Usage: python generate_portt5_features_pdb.py <protein_folder>")
        sys.exit(1)

    protein_directory = sys.argv[1]
    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    process_protein(protein_directory)


if __name__ == "__main__":
    main()