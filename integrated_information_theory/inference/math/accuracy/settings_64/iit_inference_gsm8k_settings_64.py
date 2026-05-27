from integrated_information_theory.inference.integrated_information_inference import integrated_information_inference
from integrated_information_theory.datasets.math.gsm8k_dataset import gsm8k_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.accuracy.inference_accuracy_logger import inference_accuracy_logger
from integrated_information_theory.integrated_information import integrated_information
from integrated_information_theory.config.integrated_information_config import integrated_information_config
from integrated_information_theory.enums_class import ii_phi_type_enum, ii_calculation_type_enum, tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum

class iit_inference_gsm8k_settings_64(integrated_information_inference): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        super().__init__(model_name, peft_checkpoint_path)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            self.dataset = gsm8k_dataset(config)
        return self.dataset

    
    def get_iit_calculator(self):
        return None

    def get_logger(self, run_number = 0):
        if self.logger is None:
            self.logger = inference_accuracy_logger(log_file_name = f'integrated_information_theory/inference/math/accuracy/settings_64/run_{run_number}/settings_64_gsm8k_full.csv')

        return self.logger



# for run_number in range(1,6):
#     print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
#     t = iit_inference_gsm8k_settings_64('/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_64/checkpoint-500-HF')
#     t.get_logger(run_number=run_number)
#     t.calculate_accuracy_vllm()
#     print(f'{'*' * 210}')

for run_number in range(1,6):
    print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
    t = iit_inference_gsm8k_settings_64('/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_64/checkpoint-500-HF')
    t.get_logger(run_number=run_number)
    t.calculate_entropy(t.get_logger().get_log_file_name())
    print(f'{'*' * 210}')
