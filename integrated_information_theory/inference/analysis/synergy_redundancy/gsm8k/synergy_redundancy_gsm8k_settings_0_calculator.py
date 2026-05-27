from integrated_information_theory.inference.analysis.synergy_redundancy.synergy_redundancy_calculator import synergy_redundancy_calculator


class synergy_redundancy_gsm8k_settings_0_calculator(synergy_redundancy_calculator): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        super().__init__(model_name, peft_checkpoint_path)

    def get_log_file_name(self):
        return './integrated_information_theory/inference/math/accuracy/settings_0/settings_0_gsm8k_full.csv'


c = synergy_redundancy_gsm8k_settings_0_calculator('/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', )
c.calculate()