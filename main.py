from flask import Flask, request, jsonify
import logging
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# =========================
# CONFIG
# =========================
NUM_SAMPLES = 500
NUM_EPOCHS = 20
INPUT_FILE = "reasoning_data.csv"

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# =========================
# DATA GENERATOR
# =========================
def generate_data(n):
    data = []
    for _ in range(n):
        a = random.randint(0, 20)
        b = random.randint(0, 20)

        label = 1 if (a * 2 + 3) > (b * 2 + 3) else 0

        data.append({
            "feature1": a,
            "feature2": b,
            "label": label
        })

    df = pd.DataFrame(data)
    df.to_csv(INPUT_FILE, index=False)
    return df

# =========================
# DATASET
# =========================
class ReasoningDataset(Dataset):
    def __init__(self, file):
        df = pd.read_csv(file)
        self.X = df[["feature1", "feature2"]].values.astype(np.float32)
        self.y = df["label"].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# =========================
# MODEL
# =========================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# =========================
# TRAINING
# =========================
def train_model():
    generate_data(NUM_SAMPLES)

    dataset = ReasoningDataset(INPUT_FILE)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Net()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0

        for X, y in dataloader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            logging.info(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model

# =========================
# LOAD MODEL (ONLY ONCE)
# =========================
print("Training model once...")
model = train_model()
model.eval()  # IMPORTANT

# =========================
# MONITORING
# =========================
class Monitoring:
    def __init__(self):
        self.update()

    def update(self):
        self.vision = random.randint(10, 100)
        self.language = random.randint(10, 100)
        self.reasoning = random.randint(10, 100)
        self.intelligence = self.vision + self.language + self.reasoning

    def report(self):
        return {
            "vision": self.vision,
            "language": self.language,
            "reasoning": self.reasoning,
            "intelligence": self.intelligence
        }

monitor = Monitoring()

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return jsonify({"message": "AGI Reasoning API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "feature1" not in data or "feature2" not in data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        a = float(data["feature1"])
        b = float(data["feature2"])
    except:
        return jsonify({"error": "Input must be numbers"}), 400

    x = torch.tensor([[a, b]], dtype=torch.float32)

    with torch.no_grad():  # IMPORTANT
        prediction = model(x).item()

    return jsonify({
        "input": [a, b],
        "prediction": prediction,
        "label": int(prediction > 0.5)
    })

@app.route("/monitor")
def monitor_status():
    monitor.update()
    return jsonify(monitor.report())

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)