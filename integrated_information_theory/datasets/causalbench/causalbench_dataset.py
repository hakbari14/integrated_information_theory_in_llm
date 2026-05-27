from integrated_information_theory.datasets.dataset_handler import dataset_handler
import re
import pandas as pd


class causalbench_dataset(dataset_handler): 

    def __init__(self, config):
        super().__init__(config)

    
    def final_answer_extraction(self, prompt, solution, target):
        last = solution[-min(600, len(solution)):]

        patterns = [
            r'(?i)answer.*?\b(yes|no)\b',            
            r'(?i)\b(Yes|No)\b',            
        ]

        for pattern in patterns:
            match = re.search(pattern, last, re.IGNORECASE | re.DOTALL)
            if not match: continue
            answer = match.group(1).capitalize()
            if answer not in ['Yes', 'No']: continue
            return answer
        
        return None

    def sample_unique_with_all_rows(self, df, column_name, n_sample, random_state=42):
        unique_values = df[column_name].dropna().unique()
        n_sample = min(n_sample, len(unique_values))

        sampled_values = pd.Series(unique_values).sample(
            n=n_sample,
            replace=False,
            random_state=random_state
        )
        
        return df[df[column_name].isin(sampled_values)]
    
