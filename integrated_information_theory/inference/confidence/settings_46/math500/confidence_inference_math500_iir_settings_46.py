from integrated_information_theory.inference.confidence.confidence_inference import confidence_inference
from integrated_information_theory.datasets.math.math_500_dataset import math_500_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger
from integrated_information_theory.intrinsic_information import intrinsic_information
from integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from integrated_information_theory.enums_class import ii_calculation_type_enum, tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum, iit_layer_type_enum, iit_threashold_type_enum

class confidence_inference_math500_iir_settings_46(confidence_inference): 

    def __init__(self, model_name):
        super().__init__(model_name)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            config.set_ratio_test_dataset_size(0.2)
            self.dataset = math_500_dataset(config)
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

    def get_logger(self, run_number = 0):
        if self.logger is None:
            self.logger = self_consistency_inference_logger(log_file_name = f'integrated_information_theory/inference/confidence/settings_46/math500/run_{run_number}/settings_46_math500_iir.csv')

        return self.logger

t = confidence_inference_math500_iir_settings_46('/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_46/checkpoint-500-HF')
t.get_logger(run_number=1)
t.calculate_accuracy_confidence()
