from integrated_information_theory.inference.integrated_information_inference import integrated_information_inference
from integrated_information_theory.datasets.code.humaneval.humaneval_dataset import humaneval_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.code.inference_code_logger import inference_code_logger


class iit_inference_humaneval_settings_83(integrated_information_inference): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        super().__init__(model_name, peft_checkpoint_path)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            self.dataset = humaneval_dataset(config)
        return self.dataset

    
    def get_iit_calculator(self):
        return None

    def get_output_file_path_code(self, run_number = 0):
        return f'integrated_information_theory/inference/code/humaneval/accuracy/run_{run_number}/humaneval_result_settings_83'

    def get_logger(self, run_number = 0):
        if self.logger is None:
            self.logger = inference_code_logger(log_file_name = f'integrated_information_theory/inference/code/humaneval/accuracy/run_{run_number}/settings_83_humaneval.csv')

        return self.logger


for run_number in range(1,6):
    print(f"{'*' * 100}  Run Number {run_number}  {'*' * 100}")
    t = iit_inference_humaneval_settings_83('hakbari/deepseek_r1_qwen_7B_adaptive_length_penalty_83')
    t.get_logger(run_number=run_number)
    t.calculate_accuracy_code(run_number)
    print(f"{'*' * 210}")
