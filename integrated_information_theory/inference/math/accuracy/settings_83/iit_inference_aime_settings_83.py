from integrated_information_theory.inference.integrated_information_inference import integrated_information_inference
from integrated_information_theory.datasets.math.aime_dataset import aime_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.accuracy.inference_accuracy_logger import inference_accuracy_logger
from integrated_information_theory.intrinsic_information import intrinsic_information
from integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from integrated_information_theory.enums_class import granularity_enum, last_layer_computation_type_enum, tpm_creation_type_enum, ii_calculation_type_enum, training_type_enum



class iit_inference_aime_settings_83(integrated_information_inference): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        super().__init__(model_name, peft_checkpoint_path)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            self.dataset = aime_dataset(config)
        return self.dataset

    
    def get_iit_calculator(self):
        config = intrinsic_information_config()
        config.set_calculation_type(ii_calculation_type_enum.SUM)
        config.set_adaptive_dim(False)
        config.set_reduced_dim(5)
        config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
        config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
        config.set_last_layer_computation_param(0.09)
        config.set_granularity(granularity_enum.TOKEN)
        return intrinsic_information(config) 

    def get_max_new_tokens(self):
        return 35000

    def get_logger(self, run_number = 0):
        if self.logger is None:
            self.logger = inference_accuracy_logger(log_file_name = f'integrated_information_theory/inference/math/accuracy/settings_83/run_{run_number}/settings_83_aime_full.csv')

        return self.logger


for run_number in range(1,6):
    print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
    t = iit_inference_aime_settings_83('/home/hr_akbari/research/integrated_information_theory_in_llm/live_logs/settings_83/checkpoint-350-HF')
    t.get_logger(run_number=run_number)
    t.calculate_accuracy_vllm()
    print(f'{'*' * 210}')

# for run_number in range(1,6):
#     print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
#     t = iit_inference_aime_settings_83('hakbari/deepseek_r1_qwen_7B_iit_intrinsic_information_83')
#     t.get_logger(run_number=run_number)
#     t.calculate_entropy(t.get_logger().get_log_file_name())
#     print(f'{'*' * 210}')
