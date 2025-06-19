import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class Predictor:
    def __init__(self):
        
        # Team Dataframe
        self.team_path = Path('data/team.csv')
        self.team_df = None

        # Machine Learning Stuff
        self.model = None
        self.scaler = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def get_data(self, path):
        self.team_df = pd.read_csv(path)

        X = self.team_df.drop(columns=["win"])
        y = self.team_df["win"]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
    
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    def evaluate(self):
        y_pred = self.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print(f"{self.__class__.__name__} Accuracy: {acc:.4f}")


        