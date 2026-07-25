import os 

from integrated_information_theory.inference.integrated_information_inference import integrated_information_inference
from integrated_information_theory.datasets.math.aime_dataset import aime_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.accuracy.inference_accuracy_logger import inference_accuracy_logger
from integrated_information_theory.logger.inference.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger
from integrated_information_theory.intrinsic_information import intrinsic_information
from integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from integrated_information_theory.enums_class import ii_calculation_type_enum, tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum, iit_layer_type_enum, iit_threashold_type_enum


class iit_inference_aime_settings_0(integrated_information_inference): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        super().__init__(model_name, peft_checkpoint_path)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            self.dataset = aime_dataset(config)
        return self.dataset

    def get_iit_calculator(self):
        if self.iit_calculator is None: 
            config = intrinsic_information_config()
            config.set_calculation_type(ii_calculation_type_enum.SUM)
            config.set_adaptive_dim(False)
            config.set_reduced_dim(5)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            config.set_granularity(granularity_enum.TOKEN)
            self.iit_calculator = intrinsic_information(config) 
            
        return self.iit_calculator

    def get_max_new_tokens(self):
        return 15000

    def get_logger(self, run_number = 0):
        if self.logger is None:
            self.logger = inference_accuracy_logger(log_file_name = f'integrated_information_theory/inference/math/accuracy/settings_0/run_{run_number}/settings_0_aime_full.csv')

        return self.logger


for run_number in range(6,11):
    print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
    t = iit_inference_aime_settings_0('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',)
    t.get_logger(run_number=run_number)
    t.calculate_accuracy_vllm()
    print(f'{'*' * 210}')

# for run_number in range(2,6):
#     print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
#     t = iit_inference_aime_settings_0('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',)
#     t.get_logger(run_number=run_number)
#     t.calculate_and_update_iit(t.get_logger().get_log_file_name())
#     print(f'{'*' * 210}')

