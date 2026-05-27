from integrated_information_theory.inference.integrated_information_inference import integrated_information_inference
from integrated_information_theory.datasets.causalbench.text_causalbench_dataset import text_causalbench_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.causalbench.inference_causalbench_logger import inference_causalbench_logger


class iit_inference_text_causalbench_settings_46(integrated_information_inference): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        super().__init__(model_name, peft_checkpoint_path)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            self.dataset = text_causalbench_dataset(config)
        return self.dataset

    
    def get_iit_calculator(self):
        return None

    def get_logger(self, run_number = 0):
        if self.logger is None:
            self.logger = inference_causalbench_logger(log_file_name = f'integrated_information_theory/inference/causalbench/text/run_{run_number}/settings_46_causalbench.csv')

        return self.logger



for run_number in range(1,6):
    print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
    t = iit_inference_text_causalbench_settings_46('/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_46/checkpoint-500-HF')
    t.get_logger(run_number=run_number)
    t.calculate_accuracy_causal_bench()
    print(f'{'*' * 210}')
