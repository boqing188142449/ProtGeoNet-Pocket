import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from Utils.get_dataset01 import GetDataset
from Models.model import CombinedModel
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import logging

# Set device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameter settings
TEST_ROOT = r"/media/6t/zhangzhi/ProtGeoNet-Pocket/Datasets/sc6k"  # Test dataset path
MODEL_DIR = r"/media/6t/zhangzhi/ProtGeoNet-Pocket/Output"  # Model folder path
BATCH_SIZE = 1
IN_FEATURES = 2304
HIDDEN_FEATURES = 1024
THRESHOLD = 4.0  # DCC determination threshold (Å)
NUM_FOLDS = 10  # 10-fold cross-validation
RESULTS_DIR = r"/media/6t/zhangzhi/ProtGeoNet-Pocket/Output"  # Result save path

# Ensure the results directory exists
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def extract_predict_pocket_centers(pred_coords, eps=5, min_samples=5):
    """
    Perform DBSCAN clustering on predicted points and return the centroid of each non-noise cluster
    """
    if len(pred_coords) == 0:
        return np.empty((0, 3))

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pred_coords)
    labels = clustering.labels_
    centers = []
    for lbl in set(labels):
        if lbl == -1:
            continue  # Noise points
        cluster_pts = pred_coords[labels == lbl]
        centers.append(cluster_pts.mean(axis=0))

    if len(centers) == 0:
        return np.array(centers)
    return np.array(centers)

def extract_predict_pocket_centers_prob(pred_coords, pred_probs=None, min_samples=5):
    """
    Perform DBSCAN clustering on predicted points, return the centroid and average probability of each non-noise cluster.
    If no valid clusters, return original points and their probabilities.
    """
    # Input validation
    if len(pred_coords) == 0:
        return np.empty((0, 3)), np.array([])

    if pred_probs is not None:
        if len(pred_probs) != len(pred_coords):
            logging.error(f"Length of pred_probs {len(pred_probs)} does not match length of pred_coords {len(pred_coords)}")
            raise ValueError("pred_probs and pred_coords length mismatch")
    else:
        pred_probs = np.zeros(len(pred_coords))  # Default probability is 0

    # DBSCAN clustering
    clustering = DBSCAN(eps=0.01, min_samples=min_samples).fit(pred_coords)
    labels = clustering.labels_

    centers = []
    center_probs = []

    # Extract centroids and probabilities of non-noise clusters
    for lbl in set(labels):
        if lbl == -1:
            continue  # Ignore noise points
        cluster_pts = pred_coords[labels == lbl]
        cluster_prob = pred_probs[labels == lbl]
        centers.append(cluster_pts.mean(axis=0))
        center_probs.append(np.mean(cluster_prob) if len(cluster_prob) > 0 else 0.0)

    # Convert to NumPy arrays
    centers = np.array(centers)
    center_probs = np.array(center_probs)

    if len(centers) == 0:
        return pred_coords, pred_probs

    return centers, center_probs

def extract_true_pocket_centers(true_coords, min_samples=5):
    """
    Perform DBSCAN clustering on true points and return the centroid of each non-noise cluster
    """
    if len(true_coords) == 0:
        return np.empty((0, 3))

    clustering = DBSCAN(eps=0.01, min_samples=min_samples).fit(true_coords)
    labels = clustering.labels_
    labels = clustering.labels_
    centers = []
    for lbl in set(labels):
        if lbl == -1:
            continue  # Noise points
        cluster_pts = true_coords[labels == lbl]
        centers.append(cluster_pts.mean(axis=0))

    if len(centers) == 0:
        return np.array(centers)
    return np.array(centers)

def calculate_dcc_precision(predict_centers, true_centers, threshold=THRESHOLD):
    """
    Calculate spatial precision based on centroids:
      hits / total_centers
    """
    if len(predict_centers) == 0:
        return 0.0
    hits = 0
    for center in predict_centers:
        dists = np.linalg.norm(true_centers - center, axis=1)
        if np.min(dists) <= threshold:
            hits += 1
    return hits / len(predict_centers)

def compute_DCA(pred_coords, ligand_coords):
    pred_coords = np.array(pred_coords)
    ligand_coords = np.array(ligand_coords)

    if pred_coords.ndim == 1:
        pred_coords = pred_coords.reshape(1, -1)
    if ligand_coords.ndim == 1:
        ligand_coords = ligand_coords.reshape(1, -1)

    if pred_coords.shape[0] == 0 or ligand_coords.shape[0] == 0:
        return np.array([np.inf])  # Return infinity to indicate invalid

    distances = cdist(pred_coords, ligand_coords)
    return np.min(distances, axis=1)

def compute_DCC(pred_coords, true_coords):
    pred_coords = np.array(pred_coords)
    true_coords = np.array(true_coords)

    if pred_coords.ndim == 1:
        pred_coords = pred_coords.reshape(1, -1)
    if true_coords.ndim == 1:
        true_coords = true_coords.reshape(1, -1)

    if pred_coords.shape[0] == 0 or true_coords.shape[0] == 0:
        return np.array([np.inf])  # Return infinity to indicate invalid

    distances = cdist(pred_coords, true_coords)
    return np.min(distances, axis=1)

def calculate_dcc_success_rate(pred_centers, true_centers, threshold=4.0):
    if len(pred_centers) == 0 or len(true_centers) == 0:
        return 0.0, len(true_centers)

    dcc = compute_DCC(pred_centers, true_centers)
    successful_predictions = np.sum(dcc <= threshold)
    return successful_predictions / len(true_centers)

def calculate_DVO(pred_coords, ligand_coords, voxel_size=1.0):
    pred_coords = np.array(pred_coords)
    ligand_coords = np.array(ligand_coords)

    if pred_coords.ndim == 1:
        pred_coords = pred_coords.reshape(1, -1)
    if ligand_coords.ndim == 1:
        ligand_coords = ligand_coords.reshape(1, -1)

    if pred_coords.shape[0] == 0 or ligand_coords.shape[0] == 0:
        return 0.0

    # Map to voxel grid coordinates
    pred_voxels = set(map(tuple, np.floor(pred_coords / voxel_size).astype(int)))
    true_voxels = set(map(tuple, np.floor(ligand_coords / voxel_size).astype(int)))

    intersection = pred_voxels & true_voxels
    union = pred_voxels | true_voxels

    return len(intersection) / len(union) if len(union) > 0 else 0.0

def calculate_dca_success_rate(pred_coords, pred_probs, ligand_coords, n, threshold=THRESHOLD):
    if len(pred_coords) == 0 or len(ligand_coords) == 0:
        return 0, 0
    if len(pred_probs) != len(pred_coords):
        print(f"Warning: Length of pred_probs {len(pred_probs)} does not match length of pred_coords {len(pred_coords)}")
        pred_probs = pred_probs[:len(pred_coords)]
    sorted_indices = np.argsort(-pred_probs)
    pred_coords = pred_coords[sorted_indices]
    dca_values = compute_DCA(pred_coords, ligand_coords)
    top_n = min(n, len(pred_coords))
    top_n_plus_2 = min(n + 2, len(pred_coords))
    top_n_success = 1 if top_n > 0 and np.any(dca_values[:top_n] <= threshold) else 0
    top_n_plus_2_success = 1 if top_n_plus_2 > 0 and np.any(dca_values[:top_n_plus_2] <= threshold) else 0
    return top_n_success, top_n_plus_2_success

def test(model, test_loader, device):
    model.eval()
    total, count = 0, 0
    dca_top_n_success, dca_top_n_plus_2_success, dcc_success_rate, dvo_success_rate = 0, 0, 0, 0

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing", ncols=100, leave=False):
            data = data.to(device)
            output = model(data)
            output = torch.sigmoid(output)
            y = data.y.view(-1, 1).float().to(device)

            true_mask = (y.view(-1) == 1)
            if not true_mask.any():  # Skip if all labels are 0
                count += 1
                continue

            mask = (output.view(-1) > 0.5)
            true_mask = (y.view(-1) == 1)

            pred_probs = output.view(-1)[mask].cpu().numpy()
            pred_coords = data.protein_pos[mask].cpu().numpy()
            true_coords = data.protein_pos[true_mask].cpu().numpy()
            ligand_coords = data.ligand_pos.cpu().numpy()

            # Cluster and get centroids
            pred_centers = extract_predict_pocket_centers(pred_coords)
            if len(pred_centers) == 0:
                pred_centers = pred_coords

            true_centers = extract_true_pocket_centers(true_coords)
            if len(true_centers) == 0:
                true_centers = true_coords

            # Calculate DCC precision for this sample
            sample_prec = calculate_dcc_precision(pred_centers, true_centers)
            dcc_success_rate += sample_prec

            # Cluster and get centroids and probabilities
            pocket_centers, prob_centers = extract_predict_pocket_centers_prob(pred_coords, pred_probs)
            if len(pocket_centers) == 0:
                pocket_centers = pred_coords
            top_n_success, top_n_plus_2_success = calculate_dca_success_rate(pocket_centers, prob_centers,
                                                                             ligand_coords,
                                                                             len(true_centers))
            dca_top_n_success += top_n_success
            dca_top_n_plus_2_success += top_n_plus_2_success

            # Calculate DVO
            dvo_success_rate += calculate_DVO(pocket_centers, true_centers, voxel_size=2.0)

        total = (len(test_loader) - count)
        tqdm.write(f"Number of skipped samples: {count}")
        tqdm.write(f"Number of evaluated samples: {total}")

        return (
            dca_top_n_success / total if total > 0 else 0,
            dca_top_n_plus_2_success / total if total > 0 else 0,
            dcc_success_rate / total if total > 0 else 0,
            dvo_success_rate / total if total > 0 else 0
        )

def test_with_cross_validation(test_root, model_dir, device):
    print("=== 10-Fold Cross-Validation Testing ===")

    # Extract dataset name
    dataset_name = os.path.basename(test_root)
    result_path = os.path.join(RESULTS_DIR, f'{dataset_name}_cross_validation_results.txt')

    # Load the entire test dataset
    test_dataset = GetDataset(test_root)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=test_dataset.collate_fn)

    fold_results = []
    # Open file to write all results
    with open(result_path, 'w') as f:
        f.write("=== 10-Fold Cross-Validation Testing ===\n\n")

        for fold in range(NUM_FOLDS):
            print(f"\n===== Fold {fold + 1}/{NUM_FOLDS} =====")

            # Load the model for the corresponding fold
            model_path = os.path.join(model_dir, f'best_model_fold_{fold + 1}.pt')
            if not os.path.exists(model_path):
                print(f"Model file does not exist: {model_path}")
                f.write(f"===== Fold {fold + 1}/{NUM_FOLDS} =====\n")
                f.write(f"Model file does not exist: {model_path}\n\n")
                continue

            # Initialize model and load weights
            model = CombinedModel(
                input_feature=IN_FEATURES,
                out_features=1,
                hidden_features=HIDDEN_FEATURES,
                dropout=0.3
            ).to(device)
            model.load_state_dict(torch.load(model_path))

            # Perform testing using the entire test set
            metrics = test(model, test_loader, device)
            fold_results.append(metrics)

            # Write results for each fold to the file
            f.write(f"===== Fold {fold + 1}/{NUM_FOLDS} =====\n")
            f.write(f"DCA Top-n Success Rate: {metrics[0]:.4f}\n")
            f.write(f"DCA Top-(n+2) Success Rate: {metrics[1]:.4f}\n")
            f.write(f"DCC Success Rate: {metrics[2]:.4f}\n")
            f.write(f"DVO Success Rate: {metrics[3]:.4f}\n\n")

            print(f"\n=== Fold {fold + 1} Test Results ===")
            print("DCA Top-n Success Rate: %.4f" % metrics[0])
            print("DCA Top-(n+2) Success Rate: %.4f" % metrics[1])
            print("DCC Success Rate: %.4f" % metrics[2])
            print("DVO Success Rate: %.4f" % metrics[3])

        # Summarize 10-fold cross-validation results
        avg_dca_top_n = np.mean([m[0] for m in fold_results])
        avg_dca_top_n_plus_2 = np.mean([m[1] for m in fold_results])
        avg_dcc = np.mean([m[2] for m in fold_results])
        avg_dvo = np.mean([m[3] for m in fold_results])

        # Write average results to file
        f.write("=== 10-Fold Cross-Validation Summary ===\n")
        f.write(f"Average DCA Top-n Success Rate: {avg_dca_top_n:.4f}\n")
        f.write(f"Average DCA Top-(n+2) Success Rate: {avg_dca_top_n_plus_2:.4f}\n")
        f.write(f"Average DCC Success Rate: {avg_dcc:.4f}\n")
        f.write(f"Average DVO Success Rate: {avg_dvo:.4f}\n")

    print(f"\nAll results saved to {result_path}")
    print("\n=== 10-Fold Cross-Validation Summary ===")
    print(f"Average DCA Top-n Success Rate: {avg_dca_top_n:.4f}")
    print(f"Average DCA Top-(n+2) Success Rate: {avg_dca_top_n_plus_2:.4f}")
    print(f"Average DCC Success Rate: {avg_dcc:.4f}")
    print(f"Average DVO Success Rate: {avg_dvo:.4f}")

if __name__ == "__main__":
    test_with_cross_validation(TEST_ROOT, MODEL_DIR, DEVICE)