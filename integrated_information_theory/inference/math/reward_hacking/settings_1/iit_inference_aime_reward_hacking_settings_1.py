from integrated_information_theory.inference.integrated_information_inference import integrated_information_inference
from integrated_information_theory.datasets.math.aime_dataset import aime_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.reward_hacking.inference_reward_hacking_logger import inference_reward_hacking_logger
from integrated_information_theory.intrinsic_information import intrinsic_information
from integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from integrated_information_theory.enums_class import ii_calculation_type_enum, tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum

class iit_inference_aime_reward_hacking_settings_1(integrated_information_inference): 

    def __init__(self, model_name):
        super().__init__(model_name)
        self.max_new_tokens = 10000

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            config.set_max_test_dataset_size(100)
            self.dataset = aime_dataset(config)
        return self.dataset

    
    def get_iit_calculator(self):
        
        if self.iit_calculator is None:
            config = intrinsic_information_config()
            config.set_calculation_type(ii_calculation_type_enum.SUM)
            config.set_adaptive_dim(True)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            config.set_granularity(granularity_enum.TOKEN)
            self.iit_calculator = intrinsic_information(config)

        return self.iit_calculator

    def get_logger(self):
        if self.logger is None:
            self.logger = inference_reward_hacking_logger(log_file_name = 'integrated_information_theory/inference/math/reward_hacking/settings_1/settings_1.csv')

        return self.logger


t = iit_inference_aime_reward_hacking_settings_1('Qwen/Qwen2.5-1.5B')
t.compute_implicit_reward_hacking()
