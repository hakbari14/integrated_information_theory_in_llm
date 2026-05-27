from integrated_information_theory.inference.integrated_information_inference import integrated_information_inference
from integrated_information_theory.datasets.math.gsm8k_dataset import gsm8k_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger


class iit_inference_gsm8k_sc_settings_37(integrated_information_inference): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        super().__init__(model_name, peft_checkpoint_path)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            self.dataset = gsm8k_dataset(config)
        return self.dataset

    
    def get_iit_calculator(self):
        return None

    def get_logger(self):
        if self.logger is None:
            self.logger = self_consistency_inference_logger(log_file_name = 'integrated_information_theory/inference/math/self_consistency/settings_37/gsm8k/settings_37_gsm8k_sc_full.csv')

        return self.logger


t = iit_inference_gsm8k_sc_settings_37('/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_37/checkpoint-1200-HF')
t.calculate_accuracy_self_consistency_vllm()
