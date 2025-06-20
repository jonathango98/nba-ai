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

        self.team_encoder = LabelEncoder()
        self.enemy_encoder = LabelEncoder()

        self.team_df['team_encoded'] = self.team_encoder.fit_transform(self.raw_team_df['team'])
        self.team_df['enemy_encoded'] = self.enemy_encoder.fit_transform(self.raw_team_df['enemy'])

        self.team_df['win'] = self.raw_team_df['win']
        self.team_df['home'] = self.raw_team_df['home']
        self.team_df['diff'] = self.raw_team_df['diff']

        X = self.team_df[['team_encoded', 'enemy_encoded', 'home']].values
        y = self.team_df[['win', 'diff']].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if method == "pytorch":
            self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
            self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
            self.y_test = torch.tensor(self.y_test, dtype=torch.float32)

    def train(self):
        class MultiOutputNet(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(input_size, 16),
                    nn.ReLU(),
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                )
                self.win_head = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
                self.diff_head = nn.Linear(16, 1)  # for diff

            def forward(self, x):
                shared = self.shared(x)
                win = self.win_head(shared)
                diff = self.diff_head(shared)  # raw output (regression)
                return win, diff
            
        self.net = MultiOutputNet(input_size=self.X_train.shape[1])

        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        loss_win = nn.BCELoss()
        loss_diff = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)

        for epoch in range(20):
            for xb, yb in train_loader:
                win_pred, diff_pred = self.net(xb)
                loss = loss_win(win_pred, yb[:, [0]]) + loss_diff(diff_pred, yb[:, [1]])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

    def predict(self, X):
        with torch.no_grad():
            win_pred, stats_pred = self.net(X)
            return win_pred.squeeze(), stats_pred

    def evaluate(self):
        with torch.no_grad():
            win_pred, stats_pred = self.net(self.X_test)
            win_binary = (win_pred > 0.5).float()
            acc = (win_binary.squeeze() == self.y_test[:, 0]).float().mean().item()

            mse = nn.MSELoss()(stats_pred, self.y_test[:, 1:]).item()
            return {"win_accuracy": acc, "stat_mse": mse}

if __name__ == "__main__":
    model = Predictor(method="pytorch")
    model.train()

    team = "OKC"
    enemy = "IND"
    home = 1

    # Encode team and enemy
    team_encoded = model.team_encoder.transform([team])[0]
    enemy_encoded = model.enemy_encoder.transform([enemy])[0]

    # Construct the new DataFrame
    X = pd.DataFrame([{
        'team_encoded': team_encoded,
        'enemy_encoded': enemy_encoded,
        'home': home,
    }]).values

    X = torch.tensor(X, dtype=torch.float32)

    win_preds, stat_preds = model.predict(X)

    print("Win Predictions:", win_preds)
    print("Diff Predictions:", stat_preds)

    scores = model.evaluate()
    print("Win Accuracy:", scores['win_accuracy'])
    print("Regression MSE:", scores['stat_mse'])
