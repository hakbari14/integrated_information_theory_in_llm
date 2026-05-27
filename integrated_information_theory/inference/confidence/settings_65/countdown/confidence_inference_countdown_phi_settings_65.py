from integrated_information_theory.inference.confidence.confidence_inference import confidence_inference
from integrated_information_theory.datasets.math.countdown_dataset import countdown_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger
from integrated_information_theory.integrated_information import integrated_information
from integrated_information_theory.config.integrated_information_config import integrated_information_config
from integrated_information_theory.enums_class import tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum, iit_layer_type_enum, iit_threashold_type_enum, ii_phi_type_enum

class confidence_inference_countdown_phi_settings_65(confidence_inference): 

    def __init__(self, model_name):
        super().__init__(model_name)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            config.set_ratio_test_dataset_size(0.2)
            self.dataset = countdown_dataset(config)
        return self.dataset

    
    def get_iit_calculator(self):
        if self.iit_calculator is None: 
            config = integrated_information_config()
            config.set_phi_type(ii_phi_type_enum.BIG_PHI)
            config.set_adaptive_dim(False)
            config.set_reduced_dim(4)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            config.set_granularity(granularity_enum.TOKEN)
            self.iit_calculator = integrated_information(config) 
            
        return self.iit_calculator

    def get_max_new_tokens(self):
        return 15000

    def get_logger(self, run_number = 0):
        if self.logger is None:
            self.logger = self_consistency_inference_logger(log_file_name = f'integrated_information_theory/inference/confidence/settings_65/countdown/run_{run_number}/settings_65_countdown_phi.csv')

        return self.logger

t = confidence_inference_countdown_phi_settings_65('/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_65/checkpoint-1200-HF')
t.get_logger(run_number=1)
t.calculate_accuracy_confidence()
