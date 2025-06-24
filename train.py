import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from Utils.get_dataset import GetDataset
from Models.model import CombinedModel
from scipy.spatial.distance import cdist

# Hyperparameters
TRAIN_BATCH_SIZE = 32  # Batch size for training
VAL_BATCH_SIZE = 1  # Batch size for validation
NUM_EPOCHS = 500
LEARNING_RATE = 1e-4
IN_FEATURES = 2304
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT_DIR = r"/media/6t/zhangzhi/ProtGeoNet-Pocket/Datasets/scPDB"
OUTPUT_DIR = r"/media/6t/zhangzhi/ProtGeoNet-Pocket/Output"  # Modified save path
os.makedirs(OUTPUT_DIR, exist_ok=True)
THRESHOLD = 4.0  # DCC determination threshold (Å)
NUM_FOLDS = 10  # 10-fold cross-validation

# ========================= Set Random Seed =========================
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========================= DCC Related Functions =========================
def compute_DCC(pred_coords, true_coords):
    pred_coords = np.array(pred_coords)
    true_coords = np.array(true_coords)
    if pred_coords.ndim == 1:
        pred_coords = pred_coords.reshape(1, -1)
    if true_coords.ndim == 1:
        true_coords = true_coords.reshape(1, -1)
    if pred_coords.shape[0] == 0 or true_coords.shape[0] == 0:
        return np.array([np.inf])
    distances = cdist(pred_coords, true_coords)
    return np.min(distances, axis=1)

def calculate_dcc_metrics(pred_coords, true_coords, threshold=THRESHOLD):
    """
    Calculate TP, FP, FN at the residue level:
    - TP: Predicted residue with DCC ≤ 4 Å to any true residue
    - FP: Predicted residue with DCC > 4 Å to all true residues
    - FN: True residues not predicted (no predicted residue with DCC ≤ 4 Å to the true residue)
    """
    tp, fp, fn = 0, 0, 0
    if len(pred_coords) == 0:
        fn = len(true_coords) if len(true_coords) > 0 else 0  # All true residues are not predicted
        return tp, fp, fn
    if len(true_coords) == 0:
        fp = len(pred_coords)  # All predicted residues are incorrect
        return tp, fp, fn

    # Calculate distances from predicted residues to the nearest true residue
    dcc_pred_to_true = compute_DCC(pred_coords, true_coords)
    for d in dcc_pred_to_true:
        if d <= threshold:
            tp += 1
        else:
            fp += 1

    # Calculate distances from true residues to the nearest predicted residue
    dcc_true_to_pred = compute_DCC(true_coords, pred_coords)
    fn = np.sum(dcc_true_to_pred > threshold)  # Number of true residues not predicted
    return tp, fp, fn

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    bar = tqdm(train_loader, ascii=True)
    for data in bar:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        y = data.y.view(-1, 1).float().to(device)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        bar.set_description(f"loss: {loss.item():.4f}")

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    with torch.no_grad():
        for data in tqdm(val_loader, ascii=True):
            data = data.to(device)
            outputs = model(data)
            y = data.y.view(-1, 1).float().to(device)
            outputs_sigmoid = torch.sigmoid(outputs)

            # Extract predicted and true residue coordinates
            mask = (outputs_sigmoid.view(-1) > 0.5)
            true_mask = (y.view(-1) == 1)
            pred_coords = data.protein_pos[mask].cpu().numpy()
            true_coords = data.protein_pos[true_mask].cpu().numpy()

            # Calculate DCC metrics at the residue level
            tp, fp, fn = calculate_dcc_metrics(pred_coords, true_coords)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    # Calculate DCC-based Precision, Recall, F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def train_model(root_dir, batch_size, num_epochs, device):
    dataset = GetDataset(root_dir)
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    # Create a file to save validation results for each fold
    metrics_file = os.path.join(OUTPUT_DIR, 'fold_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Fold Validation Metrics\n")
        f.write("======================\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        model_path = os.path.join(OUTPUT_DIR, f'best_model_fold_{fold + 1}.pt')
        if os.path.exists(model_path):
            print(f"\n===== Fold {fold + 1}/{NUM_FOLDS} already exists. Skipping... =====")
            continue
        print(f"\n===== Fold {fold + 1}/{NUM_FOLDS} =====")

        # Split into training and validation sets
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                  collate_fn=dataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_fn)

        # Calculate pos_weight
        pos_count = sum(torch.sum(dataset[i].y == 1).item() for i in train_idx)
        neg_count = sum(torch.sum(dataset[i].y == 0).item() for i in train_idx)
        pos_weight = torch.tensor([neg_count / (pos_count + 1e-6)], device=device)

        # Initialize model, optimizer, and loss function
        model = CombinedModel(input_feature=IN_FEATURES, out_features=1, hidden_features=1024, dropout=0.3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_f1 = 0
        best_epoch = 0
        not_improve_epochs = 0
        stop_count = 20

        print('Learning rate:', optimizer.param_groups[0]['lr'])

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_epoch(model, train_loader, optimizer, criterion, device)
            val_metric = validate_epoch(model, val_loader, criterion, device)

            # val_metric contains: (dcc_precision, dcc_recall, dcc_f1)
            print('Valid - Precision: %.4f,  Recall: %.4f, F1: %.4f' % tuple(val_metric))

            # Save the best model based on DCC F1
            if val_metric[2] > best_f1:  # val_metric[2] is dcc_f1
                best_f1 = val_metric[2]
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'best_model_fold_{fold + 1}.pt'))
                print(f"✓ Improved F1: {best_f1:.4f} at epoch {best_epoch}")
                not_improve_epochs = 0
            else:
                not_improve_epochs += 1
                print(f"No improvement. Best F1: {best_f1:.4f}")
                if not_improve_epochs >= stop_count:
                    print("Early stopping.")
                    break

        # Load and validate the final model
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'best_model_fold_{fold + 1}.pt')))
        val_metric = validate_epoch(model, val_loader, criterion, device)
        print(f"\n========== Fold {fold + 1} Final Validation Result ==========")
        print(" Precision: %.4f,  Recall: %.4f,  F1: %.4f" % tuple(val_metric))
        fold_results.append(val_metric)

        # Save final validation results for each fold
        with open(metrics_file, 'a') as f:
            f.write(f"Fold {fold + 1} Final: Precision: {val_metric[0]:.4f}, Recall: {val_metric[1]:.4f}, F1: {val_metric[2]:.4f}\n")
            f.write("======================\n")

    # Summarize 10-fold cross-validation results
    avg_precision = np.mean([m[0] for m in fold_results])
    avg_recall = np.mean([m[1] for m in fold_results])
    avg_f1 = np.mean([m[2] for m in fold_results])
    print("\n========== 10-Fold Cross-Validation Summary ==========")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1: {avg_f1:.4f}")

    # Save average results to file
    with open(metrics_file, 'a') as f:
        f.write("\n10-Fold Cross-Validation Summary\n")
        f.write("================================\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
        f.write(f"Average F1: {avg_f1:.4f}\n")
        f.write("================================\n")

if __name__ == "__main__":
    train_model(ROOT_DIR, TRAIN_BATCH_SIZE, NUM_EPOCHS, DEVICE)