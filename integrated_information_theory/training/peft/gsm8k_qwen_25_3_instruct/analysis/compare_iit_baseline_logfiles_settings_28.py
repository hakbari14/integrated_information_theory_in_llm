from integrated_information_theory.training.analysis.compare_iit_baseline_logfiles_base import CompareBaselineAndIIT
import pandas as pd


class CompareBaselineAndIITSetting_28(CompareBaselineAndIIT):
    def __init__(self):
        super().__init__()

        self.converge_step_baseline = 1200
        self.converge_step_iit = 1200
        self.logfiles_path = './integrated_information_theory/training/peft/gsm8k_qwen_25_3_instruct/logs'

        self.eval_baseline_path_file = f'{self.logfiles_path}/settings_4/settings_4.csv'
        self.eval_iit_csv_path_file = f'{self.logfiles_path}/settings_28/settings_28.csv'
        self.compare_csv_path_file = f'{self.logfiles_path}/settings_28/settings_28_compare.csv'


setting_28 = CompareBaselineAndIITSetting_28()
setting_28.compare_and_write()
setting_28.show_statistics()