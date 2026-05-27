from integrated_information_theory.training.analysis.compare_iit_baseline_logfiles_base import CompareBaselineAndIIT
import pandas as pd


class CompareBaselineAndIITSetting_10(CompareBaselineAndIIT):
    def __init__(self):
        super().__init__()

        self.converge_step_baseline = 1200
        self.converge_step_iit = 1200
        self.logfiles_path = './integrated_information_theory/training/peft/gsm8k_qwen_25_3_instruct/logs'

        self.eval_baseline_path_file = f'{self.logfiles_path}/settings_4/settings_4.csv'
        self.eval_iit_csv_path_file = f'{self.logfiles_path}/settings_10/settings_10.csv'
        self.compare_csv_path_file = f'{self.logfiles_path}/settings_10/settings_10_compare.csv'


setting_10 = CompareBaselineAndIITSetting_10()
# setting_10.compare_and_write()
setting_10.show_statistics()