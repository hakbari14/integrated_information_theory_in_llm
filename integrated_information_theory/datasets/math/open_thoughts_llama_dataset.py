from integrated_information_theory.datasets.math.open_thoughts_dataset import open_thoughts_dataset

class open_thoughts_llama_dataset(open_thoughts_dataset): 

    def __init__(self, config):
        super().__init__(config)
        self.instruction = self.prompt_config.get('Math', 'open_thought_llama')        
        self.force_generate_answer_text = '\\boxed{'
