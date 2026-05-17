import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


class ExoplanetDataset(Dataset):
    """Custom PyTorch Dataset class to parse binned exoplanet time-series arrays."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # PyTorch 1D Convolutions expect an input tensor shape of (Batch, Channels, Time_Steps).
        # We unsqueeze our flat vector to introduce a singular explicit Channel dimension.
        self.X = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def load_config(config_path: str = "config.yaml") -> dict:
    """Reads the global centralized MLOps configuration profile from disk."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing at root: '{config_path}'")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(
            f"[-] Configuration file at '{config_path}' parsed as empty (None). "
            f"Please check for YAML indentation issues, syntax errors, or unaligned colons."
        )
    return cfg


def get_exoplanet_loaders(batch_size: int = None):
    """Loads parameters from config.yaml, reads cached NumPy tensors, performs

    a stratified train/test split, and returns PyTorch DataLoader objects.
    """
    # Dynamically extract parameters from the Single Source of Truth
    cfg = load_config()

    processed_dir = cfg["data"]["processed_dir"]
    test_size = cfg["data"]["test_split_ratio"]
    random_state = cfg["data"]["random_seed"]

    # Use config batch_size if none is explicitly provided during functional runtime overrides
    if batch_size is None:
        batch_size = cfg["training"]["batch_size"]

    X_path = os.path.join(processed_dir, "X_train.npy")
    y_path = os.path.join(processed_dir, "y_train.npy")

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Cached tensor matrices missing from '{processed_dir}'. Please run data_pipeline.py first."
        )

    # Ingest cached arrays from local storage
    X = np.load(X_path)
    y = np.load(y_path)

    print(f"[*] Ingested {X.shape[0]} processed stars from storage layers...")

    # Systems Optimization: Stratified splits based on parsed global configuration state parameters
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Wrap raw arrays inside our custom PyTorch Dataset tracker
    train_dataset = ExoplanetDataset(X_train, y_train)
    val_dataset = ExoplanetDataset(X_val, y_val)

    # Instantiate our stream loaders using configured parameters
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data to prevent positional bias
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep validation sequence deterministic
        drop_last=False,
    )

    print(
        f"[+] Data successfully partitioned into Train Loader ({len(train_dataset)} stars) "
        f"and Validation Loader ({len(val_dataset)} stars)."
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Smoke Test: Verify PyTorch tensor transformations run correctly with YAML configuration profiles
    try:
        train_loader, val_loader = get_exoplanet_loaders()

        # Pull a single sample batch out of our stream pipeline
        sample_features, sample_labels = next(iter(train_loader))

        print("\n[+] Dataset pipeline functional check successful!")
        print(
            f"Tensor Batch Feature Shape: {sample_features.shape} -> (Batch, Channels, Sequences)"
        )
        print(f"Tensor Batch Label Shape:   {sample_labels.shape} -> (Batch Targets)")

    except Exception as e:
        print(f"\n[-] Dataset compilation failed: {str(e)}")
