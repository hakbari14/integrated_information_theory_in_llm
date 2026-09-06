from integrated_information_theory.inference.integrated_information_inference import integrated_information_inference
from integrated_information_theory.datasets.code.humaneval.humaneval_dataset import humaneval_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger

class iit_inference_humaneval_sc_settings_83(integrated_information_inference): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        super().__init__(model_name, peft_checkpoint_path)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            self.dataset = humaneval_dataset(config)
        return self.dataset

    
    def get_iit_calculator(self):
        return None

    def get_output_file_path_code(self):
        return 'integrated_information_theory/inference/code/humaneval/self_consistency/humaneval_result_sc_settings_83'

    def get_logger(self):
        if self.logger is None:
            self.logger = self_consistency_inference_logger(log_file_name = 'integrated_information_theory/inference/code/humaneval/self_consistency/settings_83_humaneval_sc.csv')

        return self.logger


t = iit_inference_humaneval_sc_settings_83('hakbari/deepseek_r1_qwen_7B_adaptive_length_penalty_83')
t.calculate_accuracy_self_consistency_code()
