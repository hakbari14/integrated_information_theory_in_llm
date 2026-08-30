from integrated_information_theory.inference.integrated_information_inference import integrated_information_inference
from integrated_information_theory.datasets.math.countdown_dataset import countdown_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.accuracy.inference_accuracy_logger import inference_accuracy_logger
from integrated_information_theory.intrinsic_information import intrinsic_information
from integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from integrated_information_theory.enums_class import ii_calculation_type_enum, tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum


class iit_inference_countdown_settings_82(integrated_information_inference): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        super().__init__(model_name, peft_checkpoint_path)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            self.dataset = countdown_dataset(config)
        return self.dataset

    
    def get_iit_calculator(self):
        return None

    def get_max_new_tokens(self):
        return 15000

    def get_logger(self, run_number = 0):
        if self.logger is None:
            self.logger = inference_accuracy_logger(log_file_name = f'integrated_information_theory/inference/math/accuracy/settings_82/run_{run_number}/settings_82_countdown_full.csv')

        return self.logger

for run_number in range(1,6):
    print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
    t = iit_inference_countdown_settings_82('hakbari/qwen3-8b_iit_entropy_minimization_82')
    t.get_logger(run_number=run_number)
    t.calculate_accuracy_vllm()
    print(f'{'*' * 210}')

# for run_number in range(1,6):
#     print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
#     t = iit_inference_countdown_settings_82('hakbari/qwen3-8b_iit_entropy_minimization_82')
#     t.get_logger(run_number=run_number)
#     t.calculate_entropy(t.get_logger().get_log_file_name())
#     print(f'{'*' * 210}')
