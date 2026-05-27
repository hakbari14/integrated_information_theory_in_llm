from integrated_information_theory.datasets.code.code_dataset_handler import code_dataset_handler
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.enums_class import llm_pipeline_type_enum
from datasets import Dataset
import pandas as pd
import re
import textwrap


class humaneval_dataset(code_dataset_handler): 

    def __init__(self, config):
        super().__init__(config)

        df = pd.read_json('./integrated_information_theory/datasets/data/openai_humaneval/human-eval-v2-20210705.jsonl', lines=True)        
        ds = Dataset.from_pandas(df).select_columns(['task_id', 'prompt', 'entry_point', 'canonical_solution', 'test'])
        self.dataset = ds
        self.dataset = self.dataset.add_column('unique_id', range(len(self.dataset)))
        self.train_dataset = Dataset.from_dict({"prompt": [], "target": [], "problem_id" : []})
        self.test_dataset = self.dataset

    
    def code_extraction(self, completion, entry_point):
        patterns = [
            rf'(?i)```(?:[Pp]ython\n)?.*?def\s+{entry_point}.*?:\n(.*?)\n```',            
            rf'(?i)def\s+{entry_point}.*?:\n(.*?)(?:\n(?!\n*(?: |\t))|$)',            
            rf'(?i)def.*?:\n(.*?)(?:\n(?!\n*(?: |\t))|$)',            
        ]

        for pattern in patterns:
            match = re.search(pattern, completion, re.DOTALL)
            if not match: continue
            return match.group(1)
        
        return textwrap.indent(completion, ' ' * 4)

    def generate_model_prompt(self, x):
        problem_id = x["task_id"]
        prompt = self.build_prompt(x["prompt"])
        entry_point = x["entry_point"]
        canonical_solution = x["canonical_solution"]
        test = x["test"]

        r_prefix = [
            {"role": "user",
                "content": prompt
                },
        ]

        return {
                "prompt": self.tokenizer.apply_chat_template(r_prefix, tokenize=False, continue_final_message=True), 
                "target": test, 
                "entry_point": entry_point, 
                "canonical_solution": canonical_solution, 
                "problem_id": problem_id
                }


    def build_prompt(self, prompt):
        return f"""You are an expert Python programmer.

    Complete the following Python function so that it passes all tests.

    Requirements:
    - Only output valid Python code.
    - Do not include explanations or comments.
    - Follow the function signature exactly.

    {prompt}
    """


# config = dataset_config('/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60')
# config.set_pipeline_type(llm_pipeline_type_enum.INFERENCE)
# d = humaneval_dataset(config)
# train_dataset, test_dataset = d.preprocess_dataset()
# print(len(train_dataset))
# print(len(test_dataset))
