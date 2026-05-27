from integrated_information_theory.training.analysis.compare_iit_baseline_logfiles_base import CompareBaselineAndIIT
import pandas as pd


class CompareBaselineAndIITSetting_8(CompareBaselineAndIIT):
    def __init__(self):
        super().__init__()

        self.converge_step_baseline = 1200
        self.converge_step_iit = 1200
        self.logfiles_path = './integrated_information_theory/training/full_fine_tuning/logs'

        self.eval_baseline_path_file = f'{self.logfiles_path}/settings_0/settings_0.csv'
        self.eval_iit_csv_path_file = f'{self.logfiles_path}/settings_8/settings_8.csv'
        self.compare_csv_path_file = f'{self.logfiles_path}/settings_8/settings_8_compare.csv'


setting_8 = CompareBaselineAndIITSetting_8()
#setting_8.compare_and_write()
setting_8.show_statistics()