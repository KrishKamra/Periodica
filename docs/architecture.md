

---
# System Architecture Specification: Periodica 1D-CNN Ingestion & Inference Engine

This document provides the definitive architectural specification for **Periodica**, an asynchronous deep learning processing pipeline optimized for classifying periodic astronomical transit signals within stellar starlight datasets.

---

## 1. Global Data Flow Architecture

The software architecture is decoupled into three strict, asynchronous operational environments: Data Engineering, Tensor Processing, and Inversion Inference.


```


```
              [ NASA MAST / Raw CSV Ingest ]
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │          src/data_pipeline.py                │
     │  (Type Segregation & Median Imputation)       │
     └──────────────────────────────────────────────┘
                            │
                            ▼ [ Local Processed Tensor Caches ]
     ┌──────────────────────────────────────────────┐
     │              src/dataset.py                  │
     │  (Stratified Splitting & Batch Collation)     │
     └──────────────────────────────────────────────┘
                            │
                            ▼ [ Tensors: Batch, 1, 2000 ]
     ┌──────────────────────────────────────────────┐
     │               src/models.py                  │
     │     (1D-CNN Feature Spatial Halving)         │
     └──────────────────────────────────────────────┘
                            │
                            ▼ [ Evaluation Logit Array ]
     ┌──────────────────────────────────────────────┐
     │                src/train.py                  │
     │  (BCEWithLogitsLoss + Inverse Class Weights) │
     └──────────────────────────────────────────────┘
                            │
     ┌──────────────────────┴──────────────────────┐
     ▼                                             ▼

```

[ models/exoplanet_cnn_mvp.pth ]     [ docs/images/exoplanet_training_metrics.png ]
(Serialized Parameter State)         (Dual-Axis Evaluation Telemetry Plot)

```

---

## 2. Component Pipeline In-Depth Breakdown

### 2.1 The Data Engineering Engine (`src/data_pipeline.py`)
To isolate continuous planetary signatures from noisy stellar properties, data passes through a multi-stage validation pipeline:

* **Object-Oriented Type Segregation:** The pipeline isolates and extracts categorical descriptors (e.g., `koi_disposition`) into a tracking label vector ($y$) using `pandas.DataFrame.select_dtypes(include=[np.number])`. This ensures that subsequent mathematical operations process only pure numeric attributes.
* **Slicing & Dynamic Imputation:** Data sizing constraints are enforced *before* data cleansing operations occur. Missing indices or dead feature columns exposed inside the slice are replaced with column-wise medians using `np.nanmedian`:
  $$\tilde{x} = \text{median}(x)$$
  Columns that contain only invalid fields or `NaN` values are reset to a stable zero baseline (`0.0`).
* **Vector Row Standardization:** Min-Max scaling is calculated independently across each row axis to preserve variable relationships while eliminating distance variation across different star sizes:
  $$X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}} + \epsilon}$$
  *Where $\epsilon = 1\times10^{-8}$ prevents division-by-zero errors on perfectly flat stellar profiles.*

### 2.2 The Dataset Orchestrator (`src/dataset.py`)
This component converts local binary files into streaming PyTorch data structures:

* **Stratified Validation Splits:** To preserve rare target distributions across both partitions, data is divided using an 80/20 train/test split stratified on the classification target array ($y$).
* **Channel Injection:** Standard multi-dimensional input arrays tracking a sequence length of $N=2000$ elements are converted from shapes of $(B, N)$ to matching deep learning input formats:
  $$\text{Tensor Shape} \longrightarrow (B, 1, 2000)$$
  This configuration treats the 1D time-series data like a single-channel grayscale image slice, making it compatible with 1D Convolutional Neural Network architectures.

---

## 3. Deep Learning Network Topography (`src/models.py`)

The deep network structure uses a series of `Conv1d` feature extractors. It applies spatial pooling operations to reduce 2,000 incoming time-step variables into 250 deep convolutional features while capturing subtle periodic patterns.V


```


```
   Input Tensor Shape: (Batch, 1, 2000)
             │
             ▼

```

┌──────────────────────────────────────────────┐
│                  LAYER 1                     │
│  - Conv1d (Filters: 16, Kernel: 5, Pad: 2)   │
│  - BatchNorm1d (16)                          │
│  - MaxPool1d (Kernel: 2, Stride: 2)          │
└──────────────────────────────────────────────┘
│
▼ Output Layer Shape: (Batch, 16, 1000)
┌──────────────────────────────────────────────┐
│                  LAYER 2                     │
│  - Conv1d (Filters: 32, Kernel: 5, Pad: 2)   │
│  - BatchNorm1d (32)                          │
│  - MaxPool1d (Kernel: 2, Stride: 2)          │
└──────────────────────────────────────────────┘
│
▼ Output Layer Shape: (Batch, 32, 500)
┌──────────────────────────────────────────────┐
│                  LAYER 3                     │
│  - Conv1d (Filters: 64, Kernel: 5, Pad: 2)   │
│  - BatchNorm1d (64)                          │
│  - MaxPool1d (Kernel: 2, Stride: 2)          │
└──────────────────────────────────────────────┘
│
▼ Output Layer Shape: (Batch, 64, 250)
┌──────────────────────────────────────────────┐
│              FLATTEN & CLASSIFY              │
│  - Tensor Flattening (.view)                 │
│  - Linear Transformations (16000 -> 64)      │
│  - Regularization Dropout (p=0.5)            │
│  - Output Dense Classifier (64 -> 1 Logit)   │
└──────────────────────────────────────────────┘
│
▼ Output Classification Shape: (Batch,)

```

### Topographical Structural Highlights
* **Batch Normalization (`nn.BatchNorm1d`):** Applied right after each convolution step to minimize internal covariate shift across training batches. This stabilizes optimization and accelerates convergence speeds.
* **Dropout Regularization:** A 50% dropout probability parameter (`p=0.5`) is added before the final output classification layer to prevent co-dependency among hidden weights and limit model overfitting.

---

## 4. Balanced Optimization Mechanics (`src/train.py`)

To break out of the **Majority Class Trap** common in highly skewed astrophysics data collections, the optimization framework uses class frequencies to balance its calculations. The network incorporates a positive class weight ratio ($q$) calculated directly from training targets:

$$q = \frac{N_{\text{negatives}}}{N_{\text{positives}}} = 3.1718$$

This inverse value is passed as a constant modifier to the binary cross-entropy loss calculation:

$$\mathcal{L}_n = - \left[ q \cdot y_n \cdot \log \sigma(x_n) + (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right]$$

### Optimization Highlights
* **Logit-Space Processing:** The network outputs raw un-normalized values called **Logits** directly to `BCEWithLogitsLoss`, leveraging log-sum-exp stabilization shortcuts to prevent numerical overflow and underflow errors.
* **F1 Checkpoint Optimization:** The checkpoint storage logic evaluates the validation **F1-Score** after each epoch. This setup ignores training iterations that display deceptively low loss values but high false alarm counts, ensuring the model checkpointed to disk remains balanced.

---

## 5. Decoupled Production Configuration Engine (`config.yaml`)

System parameters, storage directory paths, optimization boundaries, and environment settings are managed entirely outside the source code using a centralized configuration dashboard:

```yaml
data:
  raw_csv_path: "data/raw/kepler_data.csv"
  processed_dir: "data/processed"
  sequence_length: 2000
  test_split_ratio: 0.2
  random_seed: 42

model:
  name: "Exoplanet1DCNN"
  weights_dir: "models"
  weights_filename: "exoplanet_cnn_mvp.pth"

training:
  epochs: 15
  batch_size: 64
  learning_rate: 0.0005

logging:
  output_dir: "docs/images"
  plot_filename: "exoplanet_training_metrics.png"

```


---
