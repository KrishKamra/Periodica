import torch
import torch.nn as nn


class Exoplanet1DCNN(nn.Module):
    """A high-performance 1D Convolutional Neural Network for identifying periodic

    transit signals in normalized stellar light curves.
    """

    def __init__(self):
        super(Exoplanet1DCNN, self).__init__()

        # Block 1: Feature Extraction
        # Input shape: (Batch, 1, 2000)
        self.layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Output shape: (Batch, 16, 1000)
        )

        # Block 2: Spatial/Temporal Hierarchy
        self.layer2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Output shape: (Batch, 32, 500)
        )

        # Block 3: Deep Representation
        self.layer3 = nn.Sequential(
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Output shape: (Batch, 64, 250)
        )

        # Block 4: Linear Classification Dense Layers
        # 64 channels * 250 remaining sequence steps = 16000 features
        self.fc = nn.Sequential(
            nn.Linear(64 * 250, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Prevents overfitting during Phase 1 training
            nn.Linear(64, 1),  # Single raw output logit for Binary Cross Entropy Loss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Flatten the structural representations into a 1D vector per batch instance
        x = x.view(x.size(0), -1)

        logits = self.fc(x)
        return logits.squeeze(
            1
        )  # Drop extra trailing axis dimension to return shape (Batch,)


if __name__ == "__main__":
    # Smoke Test: Pass a mock data tensor through the architecture to check shapes
    try:
        model = Exoplanet1DCNN()

        # Match the sample dimensions pulled directly from your successful DataLoader run
        mock_batch = torch.randn(32, 1, 2000)
        output_logits = model(mock_batch)

        print("\n[+] Model architecture integrity test successful!")
        print(f"Input Matrix Shape:  {mock_batch.shape}")
        print(f"Output Logit Shape:  {output_logits.shape} -> (Matches Batch Size)")

    except Exception as e:
        print(f"\n[-] Model architecture compilation failed: {str(e)}")
