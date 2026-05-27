from integrated_information_theory.training.peft.grpo_peft_trainer import grpo_peft_trainer
from integrated_information_theory.datasets.math.aime_dataset import aime_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config

class grpo_aime_qwen_3_peft_trainer(grpo_peft_trainer): 

    def __init__(self, training_type):
        model_name = 'Qwen/Qwen2.5-3B-Instruct'
        super().__init__(model_name, training_type)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            config.set_max_prompt_length(512)
            self.dataset = aime_dataset(config)

        return self.dataset



