from integrated_information_theory.datasets.dataset_handler import dataset_handler
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.enums_class import llm_pipeline_type_enum
from datasets import Dataset
from datasets import load_dataset
import re
import pandas as pd



class consciousness_dataset(dataset_handler): 

    def __init__(self, config):
        super().__init__(config)
        
        df = pd.read_json(self.get_dataset_path(), lines=True)
        ds = Dataset.from_pandas(df).select_columns(['question', 'true_answer', 'false_answer'])
        self.dataset = ds
        self.dataset = self.dataset.add_column('unique_id', range(len(self.dataset)))
        self.train_dataset = Dataset.from_dict({"prompt": [], "target": [], "problem_id" : []})
        self.test_dataset = self.dataset

    
    def final_answer_extraction(self, prompt, solution, target):
        last = solution[-min(600, len(solution)):]
        if "</think>" in solution:
            last = solution.split("</think>", 1)[1].strip()

        patterns = [
            r'(?i)Answer.*?([AB])\b',            
            r'(?i)boxed\s*\{\s*([ABab])\s*\}',
            r'(?i)answer(?:[\s:*()]+|\bis\b|\boption\b)*([ab])',            
        ]

        for pattern in patterns:
            match = re.search(pattern, last, re.IGNORECASE | re.DOTALL)
            if not match: continue
            answer = match.group(1).upper()
            if answer not in ['A', 'B']: continue
            return answer
        
        return None

    def generate_model_prompt(self, x):
        problem_id = x["unique_id"]

        instruction_begin = "Answer the following question with only the most correct option and no extra content.\n"
        instruction_end = "\nAnswer: "
        prompt = instruction_begin + x["question"] + instruction_end

        target = x["true_answer"]

        r_prefix = [
            {"role": "user",
                "content": prompt
                },
        ]

        return {
                "prompt": self.tokenizer.apply_chat_template(r_prefix, tokenize=False, continue_final_message=True), 
                "target": target, 
                "problem_id": problem_id
                }

    def get_dataset_path(self):
        return None