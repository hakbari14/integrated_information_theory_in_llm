import pandas as pd
import re
from collections import defaultdict
import numpy as np 

class self_consistency_analysis(object):

    @staticmethod
    def calculate_accuracy_detail():
        dir_, csv_paths = self_consistency_analysis.get_csv_paths()
        grouped_acc, grouped_len = defaultdict(list), defaultdict(list)        
        for csv_path in csv_paths:
            model = self_consistency_analysis.extract_first_number(csv_path)
            acc_list, mean_list = [], []
            for run in range(1,6):
                try:
                    filepath = f'{dir_}/{csv_path}'.replace('run_', f'run_{run}')
                    df = pd.read_csv(filepath)

                    true_count = len(df[df["Accuracy"] == True])
                    row_count = len(df["Accuracy"])
                    accuracy = 100 * (true_count / row_count)
                    acc_list.append(accuracy)

                    token_count = sum(df["Token_Count"].to_list())
                    avg_length = token_count / row_count
                    
                    mean_list.append(avg_length)
                    print(f"{csv_path}: Accuracy({accuracy:.2f}), Response Length({avg_length})")
                    
                except Exception as e:
                    print(f"{csv_path}: {e}")
            
            print(f"{csv_path}: Accuracy({np.mean(acc_list):.2f} - {np.std(acc_list):.2f}), Response Length({np.mean(mean_list):.0f} - {np.std(mean_list):.0f})")
            grouped_acc[model].append(np.mean(acc_list))        
            grouped_len[model].append(np.mean(mean_list))        

        print()
        for model in grouped_acc:
            print(f'Settings{model} : Accuracy = {np.mean(grouped_acc[model]):.2f}, Mean Token Length = {np.mean(grouped_len[model]):.0f}')            

    @staticmethod
    def calculate_accuracy():
        dir_, csv_paths = self_consistency_analysis.get_csv_paths()
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(dir_+csv_path)
                true_count = len(df[df["Accuracy"] == True])
                row_count = len(df["Accuracy"])
                accuracy = 100 * (true_count / row_count)

                print(f"{csv_path}: {accuracy:.2f}")
            except Exception as e:
                print(f"{csv_path}: {e}")

    @staticmethod
    def groupy_by():
        csv_path = '/home/hr_akbari/research/LLM_PostTraining/integrated_information_theory/inference/math/self_consistency/settings_0/gsm8k/settings_0_gsm8k_sc_full_samples.csv'
        df = pd.read_csv(csv_path)
        pd.set_option('display.max_rows', None)
        df_result = df.groupby(['Sample_ID', 'Final_Answer']).size().reset_index()
        filename = csv_path.replace('_samples.csv', '_result.csv')
        df_result.to_csv(filename, index=False)

    @staticmethod
    def extract_first_number(filename):
        match = re.search(r'\d+\.\d+|\d+', filename)
        if not match:
            return None
        
        return float(match.group()) if '.' in match.group() else int(match.group())

    @staticmethod
    def get_csv_paths():
        dir_ = '/home/hr_akbari/research/LLM_PostTraining/integrated_information_theory/inference/math/self_consistency'
        csv_paths = [
                        #  'settings_0/math500/settings_0_math500_full.csv',
                        #  'settings_37/math500/settings_37_math500_full.csv',
                        #  'settings_51/math500/settings_51_math500_full.csv',
                        #  'settings_46/math500/settings_46_math500_full.csv',
                        #  'settings_64/math500/settings_64_math500_full.csv',
                        #  'settings_65/math500/settings_65_math500_full.csv',

                        #  'settings_0/gpqa/settings_0_gpqa_full.csv', 
                        #  'settings_37/gpqa/settings_37_gpqa_full.csv',
                        #  'settings_51/gpqa/settings_51_gpqa_full.csv',
                        #  'settings_46/gpqa/settings_46_gpqa_full.csv', 
                        #  'settings_64/gpqa/settings_64_gpqa_full.csv', 
                        #  'settings_65/gpqa/settings_65_gpqa_full.csv', 

                        'settings_0/countdown/run_/settings_0_countdown_full.csv', 
                        'settings_37/countdown/run_/settings_37_countdown_full.csv',
                        'settings_51/countdown/run_/settings_51_countdown_full.csv',
                        'settings_46/countdown/run_/settings_46_countdown_full.csv', 
                        'settings_64/countdown/run_/settings_64_countdown_full.csv', 
                        'settings_65/countdown/run_/settings_65_countdown_full.csv', 

                        # 'settings_0/aime/settings_0_aime_sc_full.csv', 
                        # 'settings_37/aime/settings_37_aime_sc_full.csv',
                        # 'settings_51/aime/settings_51_aime_sc_full.csv',
                        # 'settings_46/aime/settings_46_aime_sc_full.csv', 
                        # 'settings_64/aime/settings_64_aime_sc_full.csv', 
                        # 'settings_65/aime/settings_65_aime_sc_full.csv', 

                        # 'settings_0/gsm8k/settings_0_gsm8k_sc_full.csv', 
                        # 'settings_37/gsm8k/settings_37_gsm8k_sc_full.csv',
                        # 'settings_51/gsm8k/settings_51_gsm8k_sc_full.csv',
                        # 'settings_46/gsm8k/settings_46_gsm8k_sc_full.csv', 
                        # 'settings_64/gsm8k/settings_64_gsm8k_sc_full.csv', 
                        # 'settings_65/gsm8k/settings_65_gsm8k_sc_full.csv', 
                    ]
        return dir_, csv_paths

self_consistency_analysis.calculate_accuracy_detail()
# self_consistency_analysis.groupy_by()
