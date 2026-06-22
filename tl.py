import asyncio
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import websockets


HOST = "127.0.0.1"
PORT = 8765
ARTIFACT_PATH = Path("pm25_model_bundle.pth")


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


def load_artifact() -> tuple[PM25RNN, dict]:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained artifact: {ARTIFACT_PATH}. Run `python xl.py` first."
        )

    artifact = torch.load(ARTIFACT_PATH, map_location="cpu")
    model = PM25RNN(
        cell_type=artifact["model_type"],
        input_size=len(artifact["feature_columns"]),
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    return model, artifact


async def stream_data(websocket):
    model, artifact = load_artifact()
    feature_std = artifact["feature_std"]
    feature_mean = artifact["feature_mean"]
    target_std = feature_std[-1]
    target_mean = feature_mean[-1]

    sequences = artifact["test_sequences"]
    actual_values = artifact["test_actual"]
    metrics = artifact.get("metrics", {})

    print(f"Client connected. Streaming {len(sequences)} PM2.5 predictions.")
    print(
        f"Model={artifact['model_type']} "
        f"MAE={metrics.get('mae', 0):.3f} RMSE={metrics.get('rmse', 0):.3f}"
    )

    try:
        with torch.no_grad():
            for sequence, actual in zip(sequences, actual_values):
                started_at = time.perf_counter()

                input_tensor = torch.tensor([sequence], dtype=torch.float32)
                scaled_prediction = model(input_tensor).item()
                predicted = scaled_prediction * target_std + target_mean
                latency_ms = (time.perf_counter() - started_at) * 1000

                payload = {
                    "timestamp": time.time() * 1000,
                    "model_type": artifact["model_type"],
                    "ch1_actual": float(actual),
                    "ch2_predict": float(predicted),
                    "error_abs": abs(float(actual) - float(predicted)),
                    "latency_ms": round(latency_ms, 2),
                }
                await websocket.send(json.dumps(payload))
                await asyncio.sleep(0.12)
    except websockets.ConnectionClosed:
        print("Client disconnected.")


async def main():
    async with websockets.serve(stream_data, HOST, PORT):
        print("PM2.5 inference server started.")
        print(f"WebSocket: ws://{HOST}:{PORT}")
        print("Open index.html in a browser after training finishes.")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
