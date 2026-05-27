from integrated_information_theory.inference.integrated_information_inference import integrated_information_inference
from integrated_information_theory.datasets.code.humaneval.humaneval_dataset import humaneval_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.code.inference_code_logger import inference_code_logger


class iit_inference_humaneval_settings_64(integrated_information_inference): 

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
        return 'integrated_information_theory/inference/code/humaneval/accuracy/humaneval_result_settings_64'

    def get_logger(self):
        if self.logger is None:
            self.logger = inference_code_logger(log_file_name = 'integrated_information_theory/inference/code/humaneval/accuracy/settings_64_humaneval.csv')

        return self.logger


t = iit_inference_humaneval_settings_64('/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_64/checkpoint-500-HF')
t.calculate_accuracy_code()
