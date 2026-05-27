from integrated_information_theory.inference.integrated_information_inference import integrated_information_inference
from integrated_information_theory.datasets.math.gsm8k_dataset import gsm8k_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger
from integrated_information_theory.intrinsic_information import intrinsic_information
from integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from integrated_information_theory.enums_class import ii_calculation_type_enum, tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum


import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class iit_inference_gsm8k_sc_settings_49(integrated_information_inference): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        super().__init__(model_name, peft_checkpoint_path)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            config.set_max_test_dataset_size(100)
            self.dataset = gsm8k_dataset(config)
        return self.dataset

    
    def get_iit_calculator(self):
        
        if self.iit_calculator is None:
            config = intrinsic_information_config()
            config.set_calculation_type(ii_calculation_type_enum.SUM)
            config.set_adaptive_dim(True)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            config.set_granularity(granularity_enum.CHUNK)
            config.set_chunk_size(4)
            self.iit_calculator = intrinsic_information(config)

        return self.iit_calculator

    def get_logger(self):
        if self.logger is None:
            self.logger = self_consistency_inference_logger(log_file_name = 'integrated_information_theory/inference/math/self_consistency/settings_49/settings_49_gsm8k.csv')

        return self.logger


t = iit_inference_gsm8k_sc_settings_49('/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', '/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_49/checkpoint-700')
t.calculate_accuracy_self_consistency()
