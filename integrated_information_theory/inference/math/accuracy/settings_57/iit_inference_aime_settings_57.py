from integrated_information_theory.inference.integrated_information_inference_vllm import integrated_information_inference_vllm
from integrated_information_theory.datasets.math.aime_dataset import aime_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.accuracy.inference_accuracy_logger import inference_accuracy_logger
from integrated_information_theory.intrinsic_information import intrinsic_information
from integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from integrated_information_theory.enums_class import ii_calculation_type_enum, tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum


import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class iit_inference_aime_settings_51(integrated_information_inference_vllm): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        super().__init__(model_name, peft_checkpoint_path)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            #config.set_max_test_dataset_size(100)
            self.dataset = aime_dataset(config)
        return self.dataset

    
    def get_iit_calculator(self):
        return None

    def get_logger(self):
        if self.logger is None:
            self.logger = inference_accuracy_logger(log_file_name = 'integrated_information_theory/inference/math/accuracy/settings_51/settings_51_aime_full.csv')

        return self.logger


t = iit_inference_aime_settings_51('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', '/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_51/checkpoint')
t.calculate_accuracy()


