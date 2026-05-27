from integrated_information_theory.training.full_fine_tuning.grpo_full_fine_tuning_trainer import grpo_full_fine_tuning_trainer
from integrated_information_theory.datasets.math.gsm8k_dataset import gsm8k_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config

class grpo_gsm8k_qwen_full_fine_tuning_trainer(grpo_full_fine_tuning_trainer): 

    def __init__(self, training_type):
        model_name = 'Qwen/Qwen2.5-1.5B'
        super().__init__(model_name, training_type)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            self.dataset = gsm8k_dataset(config)
            
        return self.dataset


