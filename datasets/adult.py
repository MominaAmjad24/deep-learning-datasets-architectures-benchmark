import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class AdultDataset(Dataset):
    def __init__(self, split="train"):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

        columns = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race",
            "sex", "capital-gain", "capital-loss", "hours-per-week",
            "native-country", "income"
        ]

        df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
        df = df.dropna()

        df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)

        X = df.drop("income", axis=1)
        y = df["income"]

        X = pd.get_dummies(X)

        # Train/val/test split (70/15/15)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        if split == "train":
            self.X = torch.tensor(X_train, dtype=torch.float32)
            self.y = torch.tensor(y_train.values, dtype=torch.long)
        elif split == "val":
            self.X = torch.tensor(X_val, dtype=torch.float32)
            self.y = torch.tensor(y_val.values, dtype=torch.long)
        else:
            self.X = torch.tensor(X_test, dtype=torch.float32)
            self.y = torch.tensor(y_test.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

