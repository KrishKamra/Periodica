import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import yaml

# Import modular project layers
from dataset import get_exoplanet_loaders
from models import Exoplanet1DCNN


def load_config(config_path: str = "config.yaml") -> dict:
    """Reads the global centralized MLOps configuration profile from disk."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing at root: '{config_path}'")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_metrics(preds: torch.Tensor, labels: torch.Tensor):
    """Calculates precision, recall, and f1 score for highly imbalanced binary classes."""
    preds = (preds >= 0.0).float().cpu().numpy()
    labels = labels.cpu().numpy()

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1, tp, fp, fn, tn


def train_model():
    """Orchestrates Phase 2 training with class balancing and complete multi-metric validation

    using centralized configuration parameter dashboards.
    """
    # Load configuration profile parameters dynamically
    cfg = load_config()

    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    learning_rate = cfg["training"]["learning_rate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Phase 2 training engine using device: {device}")

    # Step 1: Instantiate loaders using configuration bounds
    train_loader, val_loader = get_exoplanet_loaders(batch_size=batch_size)

    # Step 2: System Optimization - Dynamically calculate positive class weights
    train_labels = train_loader.dataset.y.numpy()
    num_negatives = np.sum(train_labels == 0)
    num_positives = np.sum(train_labels == 1)

    pos_weight_value = num_negatives / num_positives if num_positives > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    print(
        f"[*] Dataset Profile -> Negatives (No Planet): {num_negatives} | Positives (Planets): {num_positives}"
    )
    print(
        f"[+] Applying dynamic optimization loss pos_weight multiplier: {pos_weight_value:.4f}"
    )

    # Step 3: Instantiate model architectures
    model = Exoplanet1DCNN().to(device)

    # Step 4: Inject pos_weight into BCEWithLogitsLoss to counter majority class bias
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_f1 = -1.0

    # Configure production checkpointing path using YAML definitions
    checkpoint_dir = cfg["model"]["weights_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_save_path = os.path.join(checkpoint_dir, cfg["model"]["weights_filename"])

    # Step 5: Master Optimization Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        all_train_logits = []
        all_train_labels = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for features, labels in progress_bar:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            all_train_logits.append(logits.detach())
            all_train_labels.append(labels.detach())

            progress_bar.set_postfix(loss=loss.item())

        # Collate epoch performance matrices
        all_train_logits = torch.cat(all_train_logits)
        all_train_labels = torch.cat(all_train_labels)
        t_prec, t_rec, t_f1, _, _, _, _ = calculate_metrics(
            all_train_logits, all_train_labels
        )
        epoch_train_loss = train_loss / len(train_loader.dataset)

        # Step 6: Multi-Metric Validation Evaluation Phase
        model.eval()
        val_loss = 0.0
        all_val_logits = []
        all_val_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)

                logits = model(features)
                loss = criterion(logits, labels)

                val_loss += loss.item() * features.size(0)
                all_val_logits.append(logits)
                all_val_labels.append(labels)

        # Collate evaluation matrices
        all_val_logits = torch.cat(all_val_logits)
        all_val_labels = torch.cat(all_val_labels)
        v_prec, v_rec, v_f1, tp, fp, fn, tn = calculate_metrics(
            all_val_logits, all_val_labels
        )
        epoch_val_loss = val_loss / len(val_loader.dataset)

        # Display advanced diagnostic analytics
        print(
            f"--> [TRAIN] Loss: {epoch_train_loss:.4f} | Prec: {t_prec:.2f} | Rec: {t_rec:.2f} | F1: {t_f1:.2f}"
        )
        print(
            f"--> [VAL]   Loss: {epoch_val_loss:.4f} | Prec: {v_prec:.2f} | Rec: {v_rec:.2f} | F1: {v_f1:.2f}"
        )
        print(
            f"--> [Matrix] True Planets Found (TP): {tp} | Missed (FN): {fn} | False Alarms (FP): {fp} | True Negatives: {tn}"
        )

        # Step 7: Save model checkpoint based on highest validation F1 score
        if v_f1 > best_val_f1 and v_f1 > 0.0:
            best_val_f1 = v_f1
            torch.save(model.state_dict(), model_save_path)
            print(
                f"[+] Saved optimal model checkpoint (F1: {best_val_f1:.4f}) to: {model_save_path}"
            )
        print("-" * 80)

    print("\n[+] Phase 2 training engine sequence completely finished!")


if __name__ == "__main__":
    train_model()
