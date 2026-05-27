from integrated_information_theory.inference.integrated_information_inference import integrated_information_inference
from integrated_information_theory.datasets.causalbench.math_causalbench_dataset import math_causalbench_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.causalbench.inference_causalbench_logger import inference_causalbench_logger


class iit_inference_math_causalbench_settings_64(integrated_information_inference): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        super().__init__(model_name, peft_checkpoint_path)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            self.dataset = math_causalbench_dataset(config)
        return self.dataset

    
    def get_iit_calculator(self):
        return None

    def get_logger(self):
        if self.logger is None:
            self.logger = inference_causalbench_logger(log_file_name = 'integrated_information_theory/inference/causalbench/math/settings_64_math_causalbench.csv')

        return self.logger


t = iit_inference_math_causalbench_settings_64('/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60','/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_64/checkpoint-500')
t.calculate_accuracy_causal_bench()
