import numpy as np
import pandas as pd
from typing import Optional, List, Dict


class iit_reward_analyzer:

    REWARD_COLUMNS = [
        "IIR_Reward_Raw",
        "IIR_Reward",
        "Phi_S_Reward_Raw",
        "Phi_S_Reward",
        "Phi_Reward_Raw",
        "Phi_Reward",
    ]

    def __init__(self, dataframe: pd.DataFrame, num_subsets: int = 1000, seed: int = 42):
        self.prompt_column = "Sample_ID"
        self.reward_columns = (self.REWARD_COLUMNS)
        self.num_subsets = num_subsets
        self.seed = seed

        self.df = dataframe.dropna(
            subset=self.reward_columns,
            how="any"
        ).copy().reset_index(drop=True)


    def _prompt_best_of_n(self, values: np.ndarray, n: int, rng: np.random.Generator) -> float:
        values = np.asarray(values, dtype=float)
        # Remove NaN / inf values
        values = values[np.isfinite(values)]
        num_samples = len(values)

        if num_samples < n:
            raise ValueError(
                f"Only {num_samples} valid samples are available, "
                f"but N={n}."
            )

        # If N is equal to all available samples,
        # there is only one possible subset.
        if num_samples == n:
            return float(np.max(values))

        subset_max_values = np.empty(self.num_subsets, dtype=float)
        for i in range(self.num_subsets):
            indices = rng.choice(num_samples, size=n, replace=False)
            subset_max_values[i] = np.max(values[indices])

        return float(np.mean(subset_max_values))

    def best_of_n_statistics(self, reward_column: str, n: int) -> Dict:
        reward_index = self.reward_columns.index(reward_column)
        rng = np.random.default_rng(self.seed + n * 1000 + reward_index)

        prompt_scores = []
        prompt_ids = []

        grouped = self.df.groupby(self.prompt_column, sort=False,)
        for prompt_id, prompt_df in grouped:
            values = prompt_df[reward_column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            if len(values) < n: continue

            score = self._prompt_best_of_n(values=values, n=n, rng=rng)

            prompt_scores.append(score)
            prompt_ids.append(prompt_id)

        if len(prompt_scores) == 0:
            raise ValueError(
                f"No prompt has at least {n} valid samples "
                f"for reward '{reward_column}'."
            )

        prompt_scores = np.asarray(prompt_scores, dtype=float)
        mean = np.mean(prompt_scores)
        if len(prompt_scores) > 1:
            std = np.std(prompt_scores, ddof=1)
            sem = std / np.sqrt(len(prompt_scores))
        else:
            std = 0.0
            sem = 0.0

        return {
            "N": n,
            "reward": reward_column,
            "mean": float(mean),
            "std": float(std),
            "sem": float(sem),
            "num_prompts": len(prompt_scores),
            "prompt_ids": prompt_ids,
            "per_prompt": prompt_scores,
        }

    def compare_rewards(self, n: int) -> pd.DataFrame:
        rows = []
        for reward_column in self.reward_columns:
            result = self.best_of_n_statistics(reward_column=reward_column, n=n)

            rows.append({
                "N": n,
                "Reward": reward_column,
                "BestOfN": result["mean"],
                "STD": result["std"],
                "SEM": result["sem"],
                "NumPrompts": result["num_prompts"],
            })

        return pd.DataFrame(rows)

    def test(self, show_std: bool = False) -> pd.DataFrame:
        n_values = [1, 2, 4, 8, 16, 32, 64]
        table_rows = []

        for n in n_values:
            row = {"N": n}
            for reward_column in self.reward_columns:
                result = self.best_of_n_statistics(reward_column=reward_column, n=n)
                row[reward_column] = result["mean"]

                if show_std:
                    row[f"{reward_column}_std"] = result["std"]

            table_rows.append(row)

        result_df = pd.DataFrame(table_rows)
        print("\nBest-of-N Reward Comparison")
        print("=" * 100)
        print(result_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print("=" * 100)
        return result_df

    def test_detailed(self) -> pd.DataFrame:
        n_values = [1, 2, 4, 8, 16, 32, 64, 128]
        rows = []

        for n in n_values:
            for reward_column in self.reward_columns:
                result = self.best_of_n_statistics(reward_column=reward_column, n=n)

                rows.append({
                    "N": n,
                    "Reward": reward_column,
                    "BestOfN": result["mean"],
                    "STD": result["std"],
                    "SEM": result["sem"],
                    "NumPrompts": result["num_prompts"],
                })

        result_df = pd.DataFrame(rows)
        print(result_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        return result_df
    
    def reward_variance_by_prompt(self, ddof: int = 1) -> pd.DataFrame:
        # Group responses by promptID and calculate
        # variance for all six rewards.
        variance_df = (
            self.df
            .groupby(self.prompt_column)[self.reward_columns]
            .var(ddof=ddof)
            .reset_index()
        )

        variance_df = variance_df.rename(
            columns={
                reward: f"{reward}_variance"
                for reward in self.reward_columns
            }
        )

        print("\nReward Variance by Prompt")
        print("=" * 120)
        print(variance_df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
        print("=" * 120)
        return variance_df    