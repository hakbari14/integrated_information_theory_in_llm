from integrated_information_theory.inference.integrated_information_inference import integrated_information_inference
from integrated_information_theory.datasets.faithfulness.post_hoc_rationalization.post_hoc_rationalization_faithfulness_dataset import post_hoc_rationalization_faithfulness_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.faithfulness.inference_faithfulness_logger import inference_faithfulness_logger


class iit_inference_faithfulness_settings_65(integrated_information_inference): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        super().__init__(model_name, peft_checkpoint_path)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            self.dataset = post_hoc_rationalization_faithfulness_dataset(config)
        return self.dataset

    
    def get_iit_calculator(self):
        return None

    def get_logger(self):
        if self.logger is None:
            self.logger = inference_faithfulness_logger(log_file_name = 'integrated_information_theory/inference/faithfulness/post_hoc_rationalization/settings_65_faithfulness_full.csv')

        return self.logger


t = iit_inference_faithfulness_settings_65('/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', '/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_65/checkpoint-1200')
t.calculate_accuracy_faithfulness()
