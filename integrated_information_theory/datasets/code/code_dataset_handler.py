from integrated_information_theory.datasets.dataset_handler import dataset_handler

class code_dataset_handler(dataset_handler): 

    def __init__(self, config):
        super().__init__(config)

    def final_answer_extraction(self, prompt, solution_str, target):
        return None



