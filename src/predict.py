import os
import torch
import numpy as np
from models import Exoplanet1DCNN


class ExoplanetPredictor:

    def __init__(self, model_weights_path: str = "models/exoplanet_cnn_mvp.pth"):
        """Initializes the predictor by loading the trained model weights."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(
                f"Trained model weights missing at '{model_weights_path}'. Please run train.py first."
            )

        self.model = Exoplanet1DCNN()
        self.model.load_state_dict(
            torch.load(model_weights_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        print(
            f"[+] Predictor engine initialized successfully using hardware: {self.device}"
        )

    def predict_probability(self, raw_flux_sequence: np.ndarray) -> float:
        """Processes a single raw 1D flux sequence and returns the mathematical

        probability of a planet transit event being present.
        """
        flux = np.nan_to_num(raw_flux_sequence, nan=0.0).astype(np.float32)

        # Fix: Structural Normalization matching data_pipeline.py precisely
        f_min = flux.min()
        f_max = flux.max()
        denominator = f_max - f_min
        if denominator == 0.0:
            denominator = 1.0

        # Replicates the exact training space mapping
        normalized_flux = (flux - f_min) / denominator

        # Enforce strict 1D single-channel tensor positioning layout: (1, 1, 2000)
        tensor_input = torch.tensor(normalized_flux, dtype=torch.float32)
        tensor_input = tensor_input.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_logit = self.model(tensor_input)
            probability = torch.sigmoid(raw_logit).item()

        return probability


if __name__ == "__main__":
    try:
        predictor = ExoplanetPredictor()

        # Load the real processed arrays from our data pipeline
        X_path = "data/processed/X_train.npy"
        y_path = "data/processed/y_train.npy"

        if not os.path.exists(X_path):
            raise FileNotFoundError(
                "Processed training arrays missing. Please run data_pipeline.py first."
            )

        X = np.load(X_path)
        y = np.load(y_path)

        # Find indices for a real planet host (1) and a non-planet host (0)
        planet_indices = np.where(y == 1)[0]
        noise_indices = np.where(y == 0)[0]

        if len(planet_indices) == 0 or len(noise_indices) == 0:
            raise ValueError("Dataset does not contain a mix of both classes.")

        # Select the first available real examples
        real_planet_star = X[planet_indices[0]]
        real_noise_star = X[noise_indices[0]]

        # Run inference through our predictor engine
        # We bypass the local min-max scaling step here since these arrays are pre-normalized
        with torch.no_grad():
            tensor_planet = (
                torch.tensor(real_planet_star, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(predictor.device)
            )
            tensor_noise = (
                torch.tensor(real_noise_star, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(predictor.device)
            )

            prob_planet = torch.sigmoid(predictor.model(tensor_planet)).item()
            prob_noise = torch.sigmoid(predictor.model(tensor_noise)).item()

        print("\n[+] Prediction Service pipeline execution successful!")
        print(
            f"--> Target A (Real Non-Planet Star) Planet Probability: {prob_noise * 100:.2f}%"
        )
        print(
            f"--> Target B (Real Confirmed Planet) Planet Probability: {prob_planet * 100:.2f}%"
        )

    except Exception as e:
        print(f"\n[-] Prediction loop execution failed: {str(e)}")
