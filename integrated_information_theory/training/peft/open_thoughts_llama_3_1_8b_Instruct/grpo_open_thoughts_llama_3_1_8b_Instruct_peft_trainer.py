from integrated_information_theory.training.peft.grpo_peft_trainer import grpo_peft_trainer
from integrated_information_theory.datasets.math.open_thoughts_llama_dataset import open_thoughts_llama_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.enums_class import llm_pipeline_type_enum

class grpo_open_thoughts_llama_3_1_8b_Instruct_peft_trainer(grpo_peft_trainer): 

    def __init__(self, training_type):
        model_name = '/home/hr_akbari/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
        super().__init__(model_name, training_type)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            config.set_max_completion_length(5000)
            self.dataset = open_thoughts_llama_dataset(config)
            
        return self.dataset


