from integrated_information_theory.training.peft.grpo_peft_trainer_countdown import grpo_peft_trainer_countdown 
from integrated_information_theory.datasets.math.countdown_dataset import countdown_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config

class grpo_countdown_qwen_instruct_peft_trainer(grpo_peft_trainer_countdown): 

    def __init__(self, training_type):
        model_name = 'Qwen/Qwen2.5-3B-Instruct'
        super().__init__(model_name, training_type)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            self.dataset = countdown_dataset(config)
            
        return self.dataset


