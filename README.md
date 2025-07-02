# ProtGeoNet-Pocket: A Binding Site Prediction Approach Integrating Sequence, Geometry, and Graph Structure

This repository contains the source code, trained models， training sets and the test sets for ProtGeoNet-Pocket.

## Introduction

ProtGeoNet-Pocket is a novel computational model designed to predict protein binding sites with high accuracy. It combines multi-scale structural information from protein sequences, 3D geometries, and graph-based representations. The model leverages advanced protein language models (ESM-2 and ProtT5), a PointNet-based module for geometric feature extraction, and a Graph Isomorphism Network (GIN) for topological learning, achieving state-of-the-art performance in binding site prediction.This repository contains the implementation of the ProtGeoNet-Pocket model.

## Dataset

The model was trained and evaluated on the following datasets:

- **scPDB (v2017)**: 17,594 protein-ligand pairs for training.(http://bioinfo-pharma.u-strasbg.fr/scPDB/)
- **Independent Test Sets**:
  - COACH420  (https://github.com/rdk/p2rank-datasets/tree/master/coach420)
  - HOLO4K  (https://github.com/rdk/p2rank-datasets/tree/master/holo4k)
  - SC6K (https://github.com/devalab/DeepPocket)
  - PDBbind (v2020) (http://www.pdbbind.org.cn/download.php)
  - ApoHolo (apo and holo subsets) (http://biomine.cs.vcu.edu/datasets/BindingSitesPredictors/main.html)

Binding sites are defined using a distance-based approach (residues within 10 Å of ligand heavy atoms).

## Performance

ProtGeoNet-Pocket achieves superior performance across multiple metrics:

- **scPDB Validation (10-fold CV)**: Average F1-score of 72.87% (Precision: 65.95%, Recall: 81.41%).
- **Independent Test Sets (10-fold CV)**:
  - COACH420: Top-n 86.76%, DCC 91.75%, DVO 65.61%
  - HOLO4K: DCC 95.92%, DVO 59.66%
  - PDBbind: Top-n 75.53%, DVO 60.32%
  - ApoHolo: Top-n 71.10%

See Table 4 in the paper for detailed results.

## Installation

- Python 3.8+
- PyTorch
- PyTorch Geometric
- NumPy, SciPy, scikit-learn
- Transformers (Hugging Face)
- ESM (Evolutionary Scale Modeling)
- Hardware: NVIDIA GPU with CUDA 11.8 support (e.g., RTX 4090)

## Dataset Preparation

The repository includes scripts to preprocess `.mol2` and `.pdb` files from various datasets to generate necessary inputs for the model, including coordinates, sequence embeddings, binding labels, graph edges, and fused features.

### Processing `.mol2` Files

The following scripts process `.mol2` files from datasets like scPDB, PDBbind, and sc6k:

- **`process_scpdb.py`**: Copies `protein.mol2` and `ligand.mol2` files to a new directory structure.

  ```bash
  python ProtGeoNet-Pocket\Processed\process_scpdb.py <scPDB_folder> <output_folder>
  ```

- **`process_PDBbind.py`**: Extracts `protein.mol2` and `ligand.mol2` files from PDBbind dataset folders.

  ```bash
  python ProtGeoNet-Pocket\Processed\process_PDBbind.py <PDBbind_folder> <output_folder>
  ```

- **`process_sc6k.py`**: Copies and renames `protein.mol2` and `ligand.mol2` files from sc6k dataset folders.

  ```bash
  python ProtGeoNet-Pocket\Processed\process_sc6k.py <sc6k_folder> <output_folder>
  ```

- **`generate_coordinates.py`**: Extracts residue centroids and ligand heavy atom coordinates from `.mol2` files, saving them as `.npy` files.

  ```bash
  python generate_coordinates.py <protein_folder>
  ```

- **`generate_esm_features.py`**: Generates ESM-2 embeddings (1280D) from protein sequences extracted from `.mol2` files.

  ```bash
  python ProtGeoNet-Pocket\Utils\mol2_file\generate_esm_features.py <protein_folder>
  ```

- **`generate_prott5_features.py`**: Generates ProtT5 embeddings (1024D) from protein sequences extracted from `.mol2` files.

  ```bash
  python ProtGeoNet-Pocket\Utils\mol2_file\generate_prott5_features.py <protein_folder>
  ```

- **`generate_binding_labels.py`**: Labels residues as binding sites if within 10 Å of ligand heavy atoms in `.mol2` files, saving labels as `.npy` files.

  ```bash
  python ProtGeoNet-Pocket\Utils\mol2_file\generate_binding_labels.py <protein_folder>
  ```

- **`generate_graph_edges.py`**: Generates graph edges and features (20D) based on a 4 Å proximity threshold between residue centroids in `.mol2` files.

  ```bash
  python ProtGeoNet-Pocket\Utils\mol2_file\generate_graph_edges.py <protein_folder>
  ```

### Processing `.pdb` Files

The following scripts process `.pdb` files from datasets like COACH420, HOLO4K, and ApoHolo:

- **`process_coach420_holo4k.py`**: Extracts protein and ligand atoms from `.pdb` files, saving them as `protein.pdb` and `ligand.pdb`.

  ```bash
  python ProtGeoNet-Pocket\Processed\process_coach420_holo4k.py <protein_folder> <output_folder>
  ```

- **`process_apoholo.py`**: Extracts protein and ligand atoms from ApoHolo `.pdb` files, saving them as `protein.pdb` and `ligand.pdb`.

  ```bash
  python ProtGeoNet-Pocket\Processed\process_apoholo.py <protein_folder> <output_folder>
  ```

- **`copy_ligand_apoholo.py`**: Copies `ligand.pdb` files from source to destination folders in ApoHolo based on folder prefixes.

  ```bash
  python ProtGeoNet-Pocket\Processed\copy_ligand_apoholo.py
  ```

- **`generate_coordinates.py`**: Extracts residue centroids and ligand heavy atom coordinates from `.pdb` files, saving them as `.npy` files.

  ```bash
  python ProtGeoNet-Pocket\Utils\pdb_file\generate_coordinates.py <protein_folder>
  ```

- **`generate_esm_features.py`**: Generates ESM-2 embeddings (1280D) from protein sequences extracted from `.pdb` files.

  ```bash
  python ProtGeoNet-Pocket\Utils\pdb_file\generate_esm_features.py <protein_folder>
  ```

- **`generate_prott5_features.py`**: Generates ProtT5 embeddings (1024D) from protein sequences extracted from `.pdb` files.

  ```bash
  python ProtGeoNet-Pocket\Utils\pdb_file\generate_prott5_features.py <protein_folder>
  ```

- **`generate_binding_labels.py`**: Labels residues as binding sites if within 10 Å of ligand heavy atoms in `.pdb` files, saving labels as `.npy` files.

  ```bash
  python ProtGeoNet-Pocket\Utils\pdb_file\generate_binding_labels.py <protein_folder>
  ```

- **`generate_graph_edges.py`**: Generates graph edges and features (20D) based on a 4 Å proximity threshold between residue centroids in `.pdb` files.

  ```bash
  python ProtGeoNet-Pocket\Utils\pdb_file\generate_graph_edges.py <protein_folder>
  ```

### Feature Fusion

- **`fuse_features.py`**: Combines ESM-2 (1280D) and ProtT5 (1024D) embeddings into a single 2304D feature file per protein.

  ```bash
  python ProtGeoNet-Pocket\Utils\fuse_features.py <protein_folder>
  ```

## Training

The `train.py` script performs 10-fold cross-validation on the scPDB dataset using the following settings:

- **Hyperparameters**:
  - Batch size: 32 (training), 1 (validation)
  - Epochs: 500
  - Learning rate: 1e-4
  - Dropout: 0.3
- **Model**: `CombinedModel` integrating PointNet and GIN.
- **Loss Function**: BCEWithLogitsLoss with pos_weight to handle class imbalance.
- **Evaluation Metrics**: Precision, Recall, F1-score based on DCC (Distance to Closest Centroid).
- **Early Stopping**: Stops after 20 epochs without F1-score improvement.
- **Output**: Saves the best model for each fold (`best_model_fold_{fold}.pt`) and metrics (`fold_metrics.txt`) in the output directory. Results are saved in `{dataset_name}_cross_validation_results.txt` in the output directory.

To train the model:

```bash
python train.py
```

## Testing

The `test.py` script evaluates the trained models on independent test sets  using 10-fold cross-validation. It computes the following metrics:

- **DCA (Distance to Closest Atom)**: Success rate for top-n and top-(n+2) predictions.
- **DCC (Distance to Closest Centroid)**: Success rate based on clustered pocket centroids.
- **DVO (Dice-like Voxel Overlap)**: Overlap between predicted and true pocket voxels.

To test the model:

```bash
python test.py
```

# Citation and contact

```
@article{
}
```

