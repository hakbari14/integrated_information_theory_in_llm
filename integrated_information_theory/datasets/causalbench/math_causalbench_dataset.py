from integrated_information_theory.datasets.causalbench.causalbench_dataset import causalbench_dataset
from datasets import Dataset
import pandas as pd

class math_causalbench_dataset(causalbench_dataset): 

    def __init__(self, config):
        super().__init__(config)
        df = pd.read_csv('./integrated_information_theory/datasets/data/causalbench/CausalBench_Math_Part.csv')
        filtered_df = super().sample_unique_with_all_rows(df, column_name='Scenario_ID', n_sample = 250)
        ds = Dataset.from_pandas(filtered_df).select_columns(['Mathematical Scenario', 'Question', 'Ground Truth', 'Scenario_ID', 'Question Type'])
        self.dataset = ds
        self.dataset = self.dataset.add_column('unique_id', range(len(self.dataset)))
        self.train_dataset = Dataset.from_dict({"prompt": [], "target": [], "problem_id" : []})
        self.test_dataset = self.dataset

    def generate_model_prompt(self, x):
        problem_id = x["unique_id"]
        senario = x["Mathematical Scenario"]
        question = x["Question"]

        instruction_begin = "Answer the following question with only the most correct option (Yes or No) and no extra content.\n"
        instruction_end = "\nAnswer: "
        prompt = instruction_begin + senario + ' ' + question + instruction_end

        target = x["Ground Truth"]
        scenario_ID = x["Scenario_ID"]
        question_type = x["Question Type"]

        r_prefix = [
            {"role": "user",
                "content": prompt
                },
        ]

        return {
                "prompt": self.tokenizer.apply_chat_template(r_prefix, tokenize=False, continue_final_message=True), 
                "target": target, 
                "problem_id": problem_id,
                "question_type": question_type,
                "scenario_ID": scenario_ID,
                }

