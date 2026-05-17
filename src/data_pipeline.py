import os
import pandas as pd
import numpy as np


class CSVLightCurvePipeline:

    def __init__(
        self,
        csv_path: str = "data/raw/kepler_data.csv",
        cache_dir: str = "data/processed",
    ):
        """Initializes a fast, high-compatibility tabular data pipeline."""
        self.csv_path = csv_path
        self.cache_dir = cache_dir

        self.X_cache_path = os.path.join(cache_dir, "X_train.npy")
        self.y_cache_path = os.path.join(cache_dir, "y_train.npy")

        os.makedirs(self.cache_dir, exist_ok=True)

    def load_and_preprocess(self, num_points: int = 2000):
        """Loads flat tabular light curves, isolates numeric columns, normalizes flux scales."""
        print(f"[*] Ingesting structured dataset from: {self.csv_path}...")

        if not os.path.exists(self.csv_path):
            print(
                "[!] CSV file not detected on disk. Generating an engineered mock dataset for testing..."
            )
            return self._generate_mock_dataset(num_points)

        # Step 1: Read the CSV dataframe cleanly
        df = pd.read_csv(self.csv_path)

        # Step 2: Extract labels cleanly
        label_col = (
            "koi_disposition" if "koi_disposition" in df.columns else df.columns[0]
        )
        print(f"[*] Isolating tracking labels from column: '{label_col}'")
        raw_labels = df[label_col].astype(str).str.lower()
        y = np.where(raw_labels.str.contains("confirmed"), 1, 0)

        # Step 3: Extract numerical telemetry columns exclusively
        print("[*] Extracting raw continuous numerical telemetry vectors...")
        numeric_df = df.select_dtypes(include=[np.number])
        X = numeric_df.values.astype(np.float32)

        # Step 4: Slice to uniform tensor sizing constraints BEFORE cleaning
        if X.shape[1] > num_points:
            X = X[:, :num_points]
        elif X.shape[1] < num_points:
            X = np.pad(X, ((0, 0), (0, num_points - X.shape[1])), mode="edge")

        # Step 5: Post-slice Imputation (Fixes NaNs introduced by slicing)
        if np.isnan(X).any():
            print(
                "[!] Detected missing data indices in feature matrix. Executing robust median imputation..."
            )
            # Calculate column-wise medians ignoring existing NaNs
            col_medians = np.nanmedian(X, axis=0)

            # If an entire column is empty, fill it with a neutral baseline of 0.0
            col_medians = np.nan_to_num(col_medians, nan=0.0)

            # Replace all NaN locations safely
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_medians, inds[1])

        # Step 6: Normalization along columns (per specific feature parameter)
        print("[*] Normalizing stellar flux metrics column-wise...")

        # Calculate min and max vertically across rows (axis=0)
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)

        range_denominator = X_max - X_min

        # Prevent division-by-zero on invariant static tracking columns
        range_denominator[range_denominator == 0.0] = 1.0

        # Scale attributes natively to independent [0.0, 1.0] distribution spaces
        X = (X - X_min) / range_denominator

        # Clip value boundaries to handle any precision rounding issues
        X = np.clip(X, 0.0, 1.0)

        # Final Fail-Safe: Force clean lingering edge calculation NaNs to 0.0
        X = np.nan_to_num(X, nan=0.0)

        # Save pristine binaries back to cache directory
        np.save(self.X_cache_path, X)
        np.save(self.y_cache_path, y)

        print(f"[+] Light curve matrices successfully cached to: {self.cache_dir}")
        return X, y

    def _generate_mock_dataset(self, num_points, num_stars=100):
        """Generates pristine synthetic light curve matrices to bypass file-system bottlenecks entirely."""
        np.random.seed(42)
        X = np.random.normal(1.0, 0.01, (num_stars, num_points))
        y = np.random.binomial(1, 0.3, num_stars)

        for i in range(num_stars):
            if y[i] == 1:
                period = 400
                depth = 0.05
                for step in range(0, num_points, period):
                    X[i, step : step + 30] -= depth

        np.save(self.X_cache_path, X)
        np.save(self.y_cache_path, y)
        print(f"[+] Synthetic training vectors generated and cached successfully.")
        return X, y


if __name__ == "__main__":
    pipeline = CSVLightCurvePipeline()

    try:
        X, y = pipeline.load_and_preprocess()
        print("\n[+] Pipeline execution completely successful!")
        print(f"Processed Features Tensor Shape: {X.shape} (Stars, Time Observations)")
        print(f"Processed Labels Tensor Shape: {y.shape} (Binary Targets)")
        print(f"Sample Normalized Flux Preview:\n{X[0, :5]}")

    except Exception as e:
        print(f"\n[-] Pipeline execution failed: {str(e)}")
