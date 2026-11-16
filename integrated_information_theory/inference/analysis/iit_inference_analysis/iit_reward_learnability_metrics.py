
import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, r2_score


class iit_reward_learnability_metrics:

    def __init__(self, y_true: NDArray[np.float64], y_prob: NDArray[np.float64]) -> None:
        self.y_true = np.asarray(y_true, dtype=np.float64).ravel()
        self.y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    def calculate_r2(self) -> float:
        if np.all(self.y_true == self.y_true[0]):
            return float("nan")

        return float(r2_score(self.y_true, self.y_prob))

    def calculate_auroc(self) -> float:
        return float(roc_auc_score(self.y_true, self.y_prob))

    def calculate_spearman(self) -> float:
        if (np.all(self.y_true == self.y_true[0]) or np.all(self.y_prob == self.y_prob[0])):
            return float("nan")

        result = spearmanr(self.y_true, self.y_prob)
        return float(result.statistic)

    def calculate_pairwise_ranking_accuracy(self) -> float:
        order = np.argsort(self.y_true, kind="mergesort")
        true_sorted = self.y_true[order]
        pred_sorted = self.y_prob[order]

        pred_values, pred_ranks = np.unique(pred_sorted, return_inverse=True)
        tree = np.zeros(len(pred_values) + 1, dtype=np.int64)

        def update(index: int) -> None:
            index += 1
            while index < len(tree):
                tree[index] += 1
                index += index & -index

        def prefix_sum(index: int) -> int:
            total = 0
            index += 1
            while index > 0:
                total += int(tree[index])
                index -= index & -index
            return total

        concordant = 0
        predicted_ties = 0
        valid_pairs = 0
        previous_count = 0

        n = len(true_sorted)
        start = 0

        while start < n:
            end = start + 1

            while (end < n and true_sorted[end] == true_sorted[start]):
                end += 1

            # Compare only against samples with lower true reward.
            for rank in pred_ranks[start:end]:
                rank = int(rank)
                lower = prefix_sum(rank - 1)
                equal = prefix_sum(rank) - lower

                concordant += lower
                predicted_ties += equal
                valid_pairs += previous_count

            # Insert this true-reward group after comparisons.
            for rank in pred_ranks[start:end]:
                update(int(rank))

            previous_count += end - start
            start = end

        if valid_pairs == 0:
            return float("nan")

        return float((concordant + 0.5 * predicted_ties) / valid_pairs)

