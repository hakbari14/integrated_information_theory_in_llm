from integrated_information_theory.training.analysis.compare_iit_baseline_logfiles_base import CompareBaselineAndIIT
import pandas as pd


class CompareBaselineAndIITSetting_3(CompareBaselineAndIIT):
    def __init__(self):
        super().__init__()

        self.converge_step_baseline = 1200
        self.converge_step_iit = 1200
        self.logfiles_path = './integrated_information_theory/training/full_fine_tuning/logs'

        self.eval_baseline_path_file = f'{self.logfiles_path}/settings_0/settings_0.csv'
        self.eval_iit_csv_path_file = f'{self.logfiles_path}/settings_3/settings_3.csv'
        self.compare_csv_path_file = f'{self.logfiles_path}/settings_3/settings_3_compare.csv'


setting_3 = CompareBaselineAndIITSetting_3()
setting_3.compare_and_write()
setting_3.show_statistics()