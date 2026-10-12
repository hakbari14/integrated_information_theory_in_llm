
import json
import ast
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline


class iit_reward_learnability_analyzer:

    REWARD_COLUMNS = [
        "IIR_Reward_Raw",
        "IIR_Reward",
        "Phi_S_Reward_Raw",
        "Phi_S_Reward",
        "Phi_Reward_Raw",
        "Phi_Reward",
    ]

    def __init__(self,
        dataframe: pd.DataFrame,
        low_high_ratio: float = 0.2,
        test_size: float = 0.2,
        seed: int = 42,
        C: float = 1.0,
        max_iter: int = 2000,
    ):
        self.df = dataframe.copy()
        self.prompt_column = "Sample_ID"
        self.representation_column = "Completion_Mean_Representation"

        self.low_high_ratio = low_high_ratio
        self.test_size = test_size
        self.seed = seed
        self.C = C
        self.max_iter = max_iter

        self.models = {}

    @staticmethod
    def parse_representation(value):
        """Convert JSON/list/array to a flattened vector."""
        if value is None:
            return None
        try:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = ast.literal_eval(value)

            arr = np.asarray(value, dtype=np.float32)
            arr = arr.reshape(-1)
            if arr.size == 0 or not np.isfinite(arr).all():
                return None

            return arr
        except (ValueError, TypeError, SyntaxError):
            return None

    def prepare_dataframe(self):
        """Clean rewards and representations."""
        df = self.df.copy()
        for col in self.REWARD_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna(subset=[self.prompt_column, self.representation_column])
        df["_representation"] = df[self.representation_column].apply(self.parse_representation)
        df = df[df["_representation"].notna()].copy()

        if df.empty:
            raise ValueError("No valid representations found.")

        return df


    def create_reward_labels(self, df, reward_column):
        data = df.dropna(subset=[reward_column, self.prompt_column]).copy()
        selected_groups = []

        for prompt_id, group in data.groupby(self.prompt_column):
            group = group.sort_values(reward_column, ascending=True)

            n = len(group)
            n_low = int(n * self.low_high_ratio)
            n_high = int(n * self.low_high_ratio)

            if n_low == 0 or n_high == 0:
                continue

            if n_low + n_high > n:
                continue

            low = group.iloc[:n_low].copy()
            high = group.iloc[-n_high:].copy()

            low["label"] = 0
            high["label"] = 1

            selected_groups.extend([low, high])

        if not selected_groups:
            raise ValueError(
                "No prompts have enough samples for "
                "low/high reward labeling."
            )

        selected = pd.concat(selected_groups, axis=0)
        selected = selected.sample(frac=1, random_state=self.seed)
        return selected.reset_index(drop=True)

    def train_single_reward(self, df, reward_column):
        data = self.create_reward_labels(df, reward_column)
        X = np.stack(data["_representation"].to_numpy())
        y = data["label"].to_numpy()

        groups = data[self.prompt_column].to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.seed)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver="liblinear",
                random_state=self.seed,
            )),
        ])

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auroc = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)

        self.models[reward_column] = model

        return {
            "Model": "Logistic_Regression",
            "Low_High_Ratio": self.low_high_ratio,
            "Reward": reward_column,
            "AUROC": auroc,
            "Accuracy": accuracy,
            "Train_Samples": len(train_idx),
            "Test_Samples": len(test_idx),
            "Train_Prompts": len(train_groups),
            "Test_Prompts": len(test_groups),
            "Low_Samples": int((y == 0).sum()),
            "High_Samples": int((y == 1).sum()),
            "Feature_Dim": X.shape[1],
        }

    def run(self):
        df = self.prepare_dataframe()
        results = []
        for reward_column in self.REWARD_COLUMNS:
            try:
                result = self.train_single_reward(df, reward_column)
                results.append(result)
            except ValueError as e:
                print(f"Skipped: {e}")

        return results

