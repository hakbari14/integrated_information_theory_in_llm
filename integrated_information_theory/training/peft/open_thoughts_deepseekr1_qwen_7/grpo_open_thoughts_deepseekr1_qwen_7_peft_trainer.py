from integrated_information_theory.training.peft.grpo_peft_trainer import grpo_peft_trainer
from integrated_information_theory.datasets.math.open_thoughts_dataset import open_thoughts_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config

class grpo_open_thoughts_deepseekr1_qwen_7_peft_trainer(grpo_peft_trainer): 

    def __init__(self, training_type):
        model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
        super().__init__(model_name, training_type)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            config.set_max_completion_length(5000)
            self.dataset = open_thoughts_dataset(config)
            
        return self.dataset


