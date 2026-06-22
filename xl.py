import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


MODEL_TYPE = "LSTM"
DATA_PATH = Path("北京PM2.5_2010-2014.csv")
ARTIFACT_PATH = Path("pm25_model_bundle.pth")
SEQUENCE_LENGTH = 24
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

FEATURE_COLUMNS = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "pm2.5"]
TARGET_COLUMN = "pm2.5"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PM25RNN(nn.Module):
    def __init__(
        self,
        cell_type: str = "LSTM",
        input_size: int = 7,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.cell_type = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if self.cell_type == "RNN":
            self.rnn_core = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        elif self.cell_type == "GRU":
            self.rnn_core = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        elif self.cell_type == "LSTM":
            self.rnn_core = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        else:
            raise ValueError("MODEL_TYPE must be one of: RNN, GRU, LSTM")

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn_core(x)
        last_step = output[:, -1, :]
        return self.fc(last_step)


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df


def build_sequences(
    values: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequences = []
    targets = []
    for idx in range(len(values) - seq_len):
        sequences.append(values[idx : idx + seq_len])
        targets.append(values[idx + seq_len, -1])
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32).reshape(-1, 1)


def train() -> None:
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    df = load_dataframe(DATA_PATH)
    values = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    split_index = int(len(values) * TRAIN_SPLIT)
    train_values = values[:split_index]
    test_values = values[split_index - SEQUENCE_LENGTH :]

    feature_mean = train_values.mean(axis=0)
    feature_std = train_values.std(axis=0)
    feature_std[feature_std == 0] = 1.0

    train_scaled = (train_values - feature_mean) / feature_std
    test_scaled = (test_values - feature_mean) / feature_std

    x_train, y_train = build_sequences(train_scaled, SEQUENCE_LENGTH)
    x_test, y_test = build_sequences(test_scaled, SEQUENCE_LENGTH)

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PM25RNN(cell_type=MODEL_TYPE, input_size=len(FEATURE_COLUMNS)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)

            prediction = model(features)
            loss = criterion(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} - train_loss: {epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        test_predictions = model(torch.from_numpy(x_test).to(device)).cpu().numpy().reshape(-1)

    target_mean = float(feature_mean[-1])
    target_std = float(feature_std[-1])
    test_actual = y_test.reshape(-1) * target_std + target_mean
    test_predictions = test_predictions * target_std + target_mean
    mae = float(np.mean(np.abs(test_actual - test_predictions)))
    rmse = float(np.sqrt(np.mean((test_actual - test_predictions) ** 2)))

    artifact = {
        "model_type": MODEL_TYPE,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "split_index": split_index,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "metrics": {"mae": mae, "rmse": rmse},
        "model_state_dict": model.state_dict(),
        "test_sequences": x_test.tolist(),
        "test_actual": test_actual.tolist(),
    }
    torch.save(artifact, ARTIFACT_PATH)

    metrics_path = ARTIFACT_PATH.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(
            {
                "model_type": MODEL_TYPE,
                "dataset": str(DATA_PATH),
                "epochs": EPOCHS,
                "sequence_length": SEQUENCE_LENGTH,
                "train_samples": int(len(x_train)),
                "test_samples": int(len(x_test)),
                "mae": mae,
                "rmse": rmse,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved model bundle to: {ARTIFACT_PATH}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Test MAE: {mae:.3f}")
    print(f"Test RMSE: {rmse:.3f}")


if __name__ == "__main__":
    train()
