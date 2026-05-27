from integrated_information_theory.training.peft.grpo_peft_trainer import grpo_peft_trainer
from integrated_information_theory.datasets.math.gsm8k_dataset import gsm8k_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config

class grpo_gsm8k_qwen_instruct_peft_trainer(grpo_peft_trainer): 

    def __init__(self, training_type):
        # model_name = 'Qwen/Qwen2.5-3B-Instruct'
        model_name = '/home/hr_akbari/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1'
        super().__init__(model_name, training_type)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            self.dataset = gsm8k_dataset(config)
            
        return self.dataset


