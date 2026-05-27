from integrated_information_theory.inference.integrated_information_inference import integrated_information_inference
from integrated_information_theory.datasets.math.aime_dataset import aime_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.accuracy.inference_accuracy_logger import inference_accuracy_logger
from integrated_information_theory.intrinsic_information import intrinsic_information
from integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from integrated_information_theory.enums_class import granularity_enum, last_layer_computation_type_enum, tpm_creation_type_enum, ii_calculation_type_enum, training_type_enum



class iit_inference_training_settings_46(integrated_information_inference): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        super().__init__(model_name, peft_checkpoint_path)

    def get_dataset(self):
        return None

    
    def get_iit_calculator(self):
        return None

    def get_max_new_tokens(self):
        return 15000

    def get_logger(self, run_number = 0):
        if self.logger is None:
            self.logger = inference_accuracy_logger(log_file_name = 'live_logs/settings_46/settings_46.csv')

        return self.logger


t = iit_inference_training_settings_46('/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_46/checkpoint-500-HF')
t.calculate_entropy(t.get_logger().get_log_file_name())
