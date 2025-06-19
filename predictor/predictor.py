import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class Predictor(nn.Module):
    def __init__(self, method: str):
        super().__init__()
        # Team Dataframe
        self.team_path = Path('../data/team.csv')

        # Load your dataset
        self.raw_team_df = pd.read_csv(self.team_path)
        self.team_df = pd.DataFrame()

        team_encoder = LabelEncoder()
        enemy_encoder = LabelEncoder()

        self.team_df['team_encoded'] = team_encoder.fit_transform(self.raw_team_df['team'])
        self.team_df['enemy_encoded'] = enemy_encoder.fit_transform(self.raw_team_df['enemy'])

        self.team_df['win'] = self.raw_team_df['win'].astype(int)

        X = self.team_df[['team_encoded', 'enemy_encoded']].values
        y = self.team_df['win'].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if method == "pytorch":
            self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(1)
            self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
            self.y_test = torch.tensor(self.y_test, dtype=torch.float32).unsqueeze(1)

    def train(self):
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)

        for epoch in range(20):
            for xb, yb in train_loader:
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

    def predict(self):
        with torch.no_grad():
            preds = self.net(self.X_test)
            return preds
    
    def evaluate(self) -> float:
        with torch.no_grad():
            preds = self.net(self.X_test)
            predicted = (preds > 0.5).float()
            accuracy = (predicted == self.y_test).float().mean()
            return accuracy.item()
        

if __name__ == "__main__":
    test = Predictor(method="pytorch")
    test.train()
    print("Prediction sample:", test.predict()[:5].squeeze().tolist())
    print("Test Accuracy:", test.evaluate())


        