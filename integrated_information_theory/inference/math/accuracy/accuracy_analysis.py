import pandas as pd
import re
from collections import defaultdict
import numpy as np 

class accuracy_analysis(object):

    @staticmethod
    def calculate_accuracy():
        dir_, csv_paths = accuracy_analysis.get_csv_paths()
        grouped_acc, grouped_len = defaultdict(list), defaultdict(list)        
        for csv_path in csv_paths:
            model = accuracy_analysis.extract_first_number(csv_path)
            acc_list, mean_list = [], []
            for run in range(6,11):
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
                    
                except Exception as e:
                    print(f"{csv_path}: {e}")
            
            print(f"{csv_path}: Accuracy({np.mean(acc_list):.2f} - {np.std(acc_list):.2f}), Response Length({np.mean(mean_list):.0f} - {np.std(mean_list):.0f})")
            grouped_acc[model].append(np.mean(acc_list))        
            grouped_len[model].append(np.mean(mean_list))        

        print()
        for model in grouped_acc:
            print(f'Settings{model} : Accuracy = {np.mean(grouped_acc[model]):.2f}, Mean Token Length = {np.mean(grouped_len[model]):.0f}')            
        
        

    @staticmethod
    def find_mean_length():
        df = pd.read_csv('./integrated_information_theory/inference/math/accuracy/settings_46/settings_46_gsm8k_full.csv')
        df_0 = pd.read_csv('./integrated_information_theory/inference/math/accuracy/settings_0/settings_0_gsm8k_full.csv')
        best_problem_id = None
        max_reduced_length = 0
        for index, row in df.iterrows():
            if df.loc[index, "Accuracy"] == False: continue
            if len(df.loc[index, "Completion"]) == 0: continue
            if df.loc[index, "Token_Count"] > 160: continue

            problem_id = df.loc[index, "Sample_ID"]
            f = df_0[df_0['Sample_ID'] == problem_id]
            if len(f) == 0: continue

            row = f.iloc[0]
            if len(row["Completion"]) == 0: continue
            
            reduced_length = 1 - len(df.loc[index, "Completion"]) / len(row["Completion"])
            if max_reduced_length < reduced_length:
                max_reduced_length = reduced_length
                best_problem_id = problem_id

        print(f'problem id = {best_problem_id}, max_reduced_length = {max_reduced_length}')    
        row_0 = df_0[df_0['Sample_ID'] == best_problem_id].iloc[0]
        row_46 = df[df['Sample_ID'] == best_problem_id].iloc[0]
        print(row_0["Completion"])
        print('*****************************************************************************')
        print(row_46["Completion"])




    @staticmethod
    def get_csv_paths():
        dir_ = './integrated_information_theory/inference/math/accuracy'
        csv_paths = [
                        'settings_0/run_/settings_0_aime_full.csv',
                        'settings_37/run_/settings_37_aime_full.csv',
                        'settings_51/run_/settings_51_aime_full.csv',
                        'settings_46/run_/settings_46_aime_full.csv',
                        # 'settings_64/run_/settings_64_aime_full.csv',
                        # 'settings_65/run_/settings_65_aime_full.csv',
                        
                        'settings_0/run_/settings_0_math500_full.csv',
                        'settings_37/run_/settings_37_math500_full.csv',
                        'settings_51/run_/settings_51_math500_full.csv',
                        'settings_46/run_/settings_46_math500_full.csv',
                        # 'settings_64/run_/settings_64_math500_full.csv',
                        # 'settings_65/run_/settings_65_math500_full.csv',
                        
                        'settings_0/run_/settings_0_gsm8k_full.csv',
                        'settings_37/run_/settings_37_gsm8k_full.csv',
                        'settings_51/run_/settings_51_gsm8k_full.csv',
                        'settings_46/run_/settings_46_gsm8k_full.csv',
                        # 'settings_64/run_/settings_64_gsm8k_full.csv',
                        # 'settings_65/run_/settings_65_gsm8k_full.csv',
                        
                        'settings_0/run_/settings_0_gpqa_full.csv',
                        'settings_37/run_/settings_37_gpqa_full.csv',
                        'settings_51/run_/settings_51_gpqa_full.csv',
                        'settings_46/run_/settings_46_gpqa_full.csv',
                        # 'settings_64/run_/settings_64_gpqa_full.csv',
                        # 'settings_65/run_/settings_65_gpqa_full.csv',

                        'settings_0/run_/settings_0_countdown_full.csv',
                        'settings_37/run_/settings_37_countdown_full.csv',
                        'settings_51/run_/settings_51_countdown_full.csv',
                        'settings_46/run_/settings_46_countdown_full.csv',
                        # 'settings_64/run_/settings_64_countdown_full.csv',
                        # 'settings_65/run_/settings_65_countdown_full.csv',
                    ]
        return dir_, csv_paths

    @staticmethod
    def extract_first_number(filename):
        match = re.search(r'\d+\.\d+|\d+', filename)
        if not match:
            return None
        
        return float(match.group()) if '.' in match.group() else int(match.group())


accuracy_analysis.calculate_accuracy()
# accuracy_analysis.find_mean_length()
