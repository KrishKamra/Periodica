import os
import yaml
import matplotlib.pyplot as plt
import numpy as np


def load_config(config_path: str = "config.yaml") -> dict:
    """Reads the global centralized MLOps configuration profile from disk."""
    if not os.path.exists(config_path):
        # Fallback directory context if running directly inside src/
        config_path = os.path.join("..", config_path)
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Load path configurations from config dashboard if present
cfg = load_config()
if cfg and "logging" in cfg:
    output_dir = cfg["logging"]["output_dir"]
    plot_filename = cfg["logging"]["plot_filename"]
    output_path = os.path.join(output_dir, plot_filename)
else:
    # Hardcoded fallback paths matching the production matrix specifications
    output_dir = "docs/images"
    output_path = os.path.join(output_dir, "exoplanet_training_metrics.png")

os.makedirs(output_dir, exist_ok=True)

# Chronological metrics extracted directly from your successful column-wise normalized logs
epochs = np.arange(1, 16)

train_loss = [
    0.5999,
    0.4805,
    0.4682,
    0.4622,
    0.4490,
    0.4385,
    0.4397,
    0.4475,
    0.4412,
    0.4344,
    0.4332,
    0.4286,
    0.4334,
    0.4240,
    0.4271,
]
val_loss = [
    0.4278,
    0.4057,
    0.4417,
    0.4354,
    0.4062,
    0.4211,
    0.3963,
    0.4252,
    0.4055,
    0.3965,
    0.3989,
    0.4184,
    0.3995,
    0.4004,
    0.4010,
]

train_f1 = [
    0.69,
    0.76,
    0.76,
    0.76,
    0.76,
    0.76,
    0.76,
    0.76,
    0.76,
    0.76,
    0.77,
    0.76,
    0.76,
    0.77,
    0.76,
]
val_f1 = [
    0.77,
    0.79,
    0.75,
    0.74,
    0.78,
    0.80,
    0.78,
    0.79,
    0.75,
    0.78,
    0.80,
    0.79,
    0.79,
    0.78,
    0.76,
]

train_prec = [
    0.57,
    0.64,
    0.64,
    0.63,
    0.64,
    0.64,
    0.63,
    0.63,
    0.64,
    0.63,
    0.64,
    0.63,
    0.63,
    0.64,
    0.63,
]
val_prec = [
    0.65,
    0.69,
    0.61,
    0.59,
    0.67,
    0.71,
    0.66,
    0.67,
    0.61,
    0.65,
    0.70,
    0.68,
    0.68,
    0.67,
    0.63,
]

train_rec = [
    0.89,
    0.93,
    0.93,
    0.94,
    0.94,
    0.94,
    0.95,
    0.95,
    0.95,
    0.95,
    0.95,
    0.95,
    0.95,
    0.95,
    0.95,
]
val_rec = [
    0.94,
    0.94,
    0.97,
    0.98,
    0.93,
    0.93,
    0.96,
    0.95,
    0.97,
    0.97,
    0.94,
    0.94,
    0.95,
    0.95,
    0.97,
]

# Initialize design layouts
plt.rcParams["font.family"] = "sans-serif"
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

fig.suptitle(
    "Periodica: 1D-CNN Model Training Diagnostics (Column-Wise Normalized)",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

# Corporate brand color coordinates
c_train = "#1f77b4"
c_val = "#ff7f0e"

# Plot Axis 1: Loss curves tracking convergence behaviors
axes[0, 0].plot(
    epochs, train_loss, label="Train", color=c_train, marker="o", linewidth=2
)
axes[0, 0].plot(
    epochs,
    val_loss,
    label="Validation",
    color=c_val,
    marker="s",
    linewidth=2,
    linestyle="--",
)
axes[0, 0].set_title("Loss Convergence Curve", fontsize=12, fontweight="semibold")
axes[0, 0].set_xlabel("Epochs")
axes[0, 0].set_ylabel("Weighted BCE Loss")
axes[0, 0].grid(True, linestyle=":", alpha=0.6)
axes[0, 0].legend()

# Plot Axis 2: F1 score curves tracking overall classification balance
axes[0, 1].plot(epochs, train_f1, label="Train", color=c_train, marker="o", linewidth=2)
axes[0, 1].plot(
    epochs,
    val_f1,
    label="Validation",
    color=c_val,
    marker="s",
    linewidth=2,
    linestyle="--",
)
axes[0, 1].set_title("F1 Optimization Curve", fontsize=12, fontweight="semibold")
axes[0, 1].set_xlabel("Epochs")
axes[0, 1].set_ylabel("F1 Score")
axes[0, 1].grid(True, linestyle=":", alpha=0.6)
axes[0, 1].legend()

# Plot Axis 3: Precision curves tracking false positive management
axes[1, 0].plot(
    epochs, train_prec, label="Train", color=c_train, marker="o", linewidth=2
)
axes[1, 0].plot(
    epochs,
    val_prec,
    label="Validation",
    color=c_val,
    marker="s",
    linewidth=2,
    linestyle="--",
)
axes[1, 0].set_title("Precision Performance Curve", fontsize=12, fontweight="semibold")
axes[1, 0].set_xlabel("Epochs")
axes[1, 0].set_ylabel("Precision (Positive Predictive Value)")
axes[1, 0].grid(True, linestyle=":", alpha=0.6)
axes[1, 0].legend()

# Plot Axis 4: Recall curves tracking how many true positive planets were found
axes[1, 1].plot(
    epochs, train_rec, label="Train", color=c_train, marker="o", linewidth=2
)
axes[1, 1].plot(
    epochs,
    val_rec,
    label="Validation",
    color=c_val,
    marker="s",
    linewidth=2,
    linestyle="--",
)
axes[1, 1].set_title("Recall Performance Curve", fontsize=12, fontweight="semibold")
axes[1, 1].set_xlabel("Epochs")
axes[1, 1].set_ylabel("Recall (Sensitivity / True Positive Rate)")
axes[1, 1].grid(True, linestyle=":", alpha=0.6)
axes[1, 1].legend()

# Formatting bounding box structures
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(output_path, dpi=300)
plt.close()

print(f"[+] Diagnostic telemetry infographic successfully saved to: {output_path}")
