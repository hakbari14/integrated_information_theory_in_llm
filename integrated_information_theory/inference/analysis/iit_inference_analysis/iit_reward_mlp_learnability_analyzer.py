import ast
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


class RewardMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)  # logits, not probabilities


class iit_reward_mlp_learnability_analyzer:
    REWARD_COLUMNS = [
        "IIR_Reward_Raw", "IIR_Reward",
        "Phi_S_Reward_Raw", "Phi_S_Reward",
        "Phi_Reward_Raw", "Phi_Reward",
    ]

    def __init__(
        self, dataframe: pd.DataFrame, low_high_ratio=0.2,
        test_size=0.2, seed=42, pca_components=128, hidden_dim=32,
        dropout=0.3, learning_rate=1e-3, weight_decay=1e-4,
        batch_size=32, max_epochs=200, patience=20, val_size=0.2,
        device=None,
    ):
        self.df = dataframe.copy()
        self.prompt_column = "Sample_ID"
        self.representation_column = "Completion_Mean_Representation"
        self.low_high_ratio = low_high_ratio
        self.test_size, self.val_size = test_size, val_size
        self.seed = seed
        self.pca_components = pca_components
        self.hidden_dim, self.dropout = hidden_dim, dropout
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = {}
        
        if not (0 < low_high_ratio < 1):
            raise ValueError("Invalid low/high ratios")
        if not (0 < test_size < 1 and 0 < val_size < 1):
            raise ValueError("test_size and val_size must be between 0 and 1")

    @staticmethod
    def parse_representation(value):
        if value is None:
            return None
        
        try:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = ast.literal_eval(value)
            arr = np.asarray(value, dtype=np.float32)
            if arr.size == 0 or not np.isfinite(arr).all():
                return None
            return arr.reshape(-1)
        except (ValueError, TypeError, SyntaxError, OverflowError):
            return None

    def prepare_dataframe(self):
        df = self.df.copy()
        for col in self.REWARD_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing reward column: {col}")
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        
        df = df.dropna(subset=[self.prompt_column, self.representation_column]).copy()
        df["_representation"] = df[self.representation_column].apply(self.parse_representation)
        df = df[df["_representation"].notna()].copy()
        return df


    def create_reward_labels(self, df, reward_column):
        data = df.dropna(subset=[reward_column]).copy()

        selected_groups = []

        for prompt_id, group in data.groupby(self.prompt_column, sort=False):
            group = group.sort_values(
                reward_column,
                ascending=True,
                kind="stable"
            )

            n = len(group)
            n_low = int(n * self.low_high_ratio)
            n_high = int(n * self.low_high_ratio)

            if (n_low < 1 or n_high < 1 or n_low + n_high > n):
                continue

            low = group.iloc[:n_low].copy()
            high = group.iloc[-n_high:].copy()

            low["label"] = 0
            high["label"] = 1

            selected_groups.extend([low, high])

        if not selected_groups:
            raise ValueError(
                "No prompts have enough samples "
                "for low/high reward labeling."
            )

        selected = pd.concat(selected_groups,ignore_index=True)
        return selected.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def _split(self, X, y, groups, fraction, seed):
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        train_idx, holdout_idx = next(splitter.split(X, y, groups))
        
        if len(set(groups[train_idx]) & set(groups[holdout_idx])):
            raise RuntimeError("Group leakage detected")
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[holdout_idx])) < 2:
            raise ValueError("Both classes are required in each split; try another seed")
        
        return train_idx, holdout_idx

    def _evaluate(self, model, X, y):
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(self.device)).cpu()
            probs = torch.sigmoid(logits).numpy()
        
        return float(roc_auc_score(y, probs)), float(accuracy_score(y, probs >= 0.5)), probs

    def train_single_reward(self, df, reward_column):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        data = self.create_reward_labels(df, reward_column)
        X = np.stack(data["_representation"].to_numpy()).astype(np.float32)
        y = data["label"].to_numpy(dtype=np.int64)
        groups = data[self.prompt_column].to_numpy()

        trainval_idx, test_idx = self._split(X, y, groups, self.test_size, self.seed)
        tr_local, val_local = self._split(X[trainval_idx], y[trainval_idx], groups[trainval_idx], self.val_size, self.seed + 1)
        train_idx, val_idx = trainval_idx[tr_local], trainval_idx[val_local]
        assert not (set(groups[train_idx]) & set(groups[val_idx]))
        assert not (set(groups[train_idx]) & set(groups[test_idx]))
        assert not (set(groups[val_idx]) & set(groups[test_idx]))

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        X_test = scaler.transform(X[test_idx])
        max_components = min(len(train_idx), X_train.shape[1])
        
        if self.pca_components > max_components:
            raise ValueError(f"PCA {self.pca_components} requires >= {self.pca_components} training samples; got {len(train_idx)}")
        
        pca = PCA(n_components=self.pca_components, svd_solver="randomized", random_state=self.seed)
        X_train = np.ascontiguousarray(pca.fit_transform(X_train), dtype=np.float32)
        X_val = np.ascontiguousarray(pca.transform(X_val), dtype=np.float32)
        X_test = np.ascontiguousarray(pca.transform(X_test), dtype=np.float32)

        model = RewardMLP(self.pca_components, self.hidden_dim, self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y[train_idx].astype(np.float32)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator().manual_seed(self.seed))
        val_x = torch.from_numpy(X_val).to(self.device)
        val_y = torch.from_numpy(y[val_idx].astype(np.float32)).to(self.device)
        best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
        
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(criterion(model(val_x), val_y).item())
            if val_loss < best_loss - 1e-5:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= self.patience:
                    break
        
        model.load_state_dict(best_state)
        auroc, accuracy, _ = self._evaluate(model, X_test, y[test_idx])
        self.models[reward_column] = {"model": model, "scaler": scaler, "pca": pca}
        
        return {
            "Model": "MLP", "Low_High_Ratio": self.low_high_ratio,
            "Reward": reward_column, "AUROC": auroc, "Accuracy": accuracy,
            "Train_Samples": len(train_idx), "Validation_Samples": len(val_idx),
            "Test_Samples": len(test_idx),
            "Train_Prompts": len(set(groups[train_idx])),
            "Validation_Prompts": len(set(groups[val_idx])),
            "Test_Prompts": len(set(groups[test_idx])),
            "Low_Samples": int((y == 0).sum()), "High_Samples": int((y == 1).sum()),
            "Feature_Dim": X.shape[1], "PCA_Dim": self.pca_components,
            "PCA_Explained_Variance": float(pca.explained_variance_ratio_.sum()),
            "Best_Epoch": best_epoch, "Validation_Loss": best_loss,
        }

    def predict_proba(self, reward_column, representations):
        components = self.models[reward_column]
        vectors = [self.parse_representation(r) for r in representations]
        if any(v is None or v.size != int(np.prod(self.expected_shape)) for v in vectors):
            raise ValueError("Invalid representation shape or values")
        X = np.stack(vectors)
        X = components["scaler"].transform(X)
        X = np.ascontiguousarray(components["pca"].transform(X), dtype=np.float32)
        model = components["model"]
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(X).to(self.device))).cpu().numpy()

    def run(self):
        df = self.prepare_dataframe()
        results = []
        for reward_column in self.REWARD_COLUMNS:
            try:
                result = self.train_single_reward(df, reward_column)
                results.append(result)
            except ValueError as exc:
                print(f"Skipped: {exc}")
        return results
