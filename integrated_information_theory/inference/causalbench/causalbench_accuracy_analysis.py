import pandas as pd
from collections import defaultdict
import numpy as np 
import re

class causalbench_accuracy_analysis(object):

    @staticmethod
    def calculate_accuracy():
        dir_, csv_paths = causalbench_accuracy_analysis.get_filenames()
        grouped_acc, grouped_len = defaultdict(list), defaultdict(list)        
        for csv_path in csv_paths:
            model = causalbench_accuracy_analysis.extract_first_number(csv_path)
            acc_list, mean_list = [], []
            for run in range(1,6):
                try:
                    filepath = f'{dir_}/{csv_path}'.replace('run_', f'run_{run}')
                    df = pd.read_csv(filepath)

                    true_count = len(df[df["Accuracy"] == True])
                    row_count =  df['Accuracy'].notna().sum()
                    accuracy = 100 * (true_count / row_count)
                    acc_list.append(accuracy)
                    
                    avg_length = df['Completion'].dropna().str.len().mean()
                    mean_list.append(avg_length)
                    
                    # print(f"run: {run} and dataset size = {len(df)} ----- {csv_path}: Accuracy({accuracy:.2f}), Response Length({avg_length:.2f})")
                except Exception as e:
                    print(f"{csv_path}: {e}")
            
            print(f"{csv_path}: Accuracy({np.mean(acc_list):.2f} - {np.std(acc_list):.2f}), Response Length({np.mean(mean_list):.0f} - {np.std(mean_list):.2f})")
            grouped_acc[model].append(np.mean(acc_list))        
            grouped_len[model].append(np.mean(mean_list))        

        print()
        for model in grouped_acc:
            print(f'Settings{model} : Accuracy = {np.mean(grouped_acc[model]):.2f}, Mean Token Length = {np.mean(grouped_len[model]):.0f}')            

    @staticmethod
    def calculate_accuracy_old():
        dir_, csv_paths = causalbench_accuracy_analysis.get_filenames()
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(dir_+csv_path)
                # df = df[(df['Question_Type'] == 'Inference from Cause to Effect with Intervention') | (df['Question_Type'] == 'Inference from Effect to Cause with Intervention')]
                true_count = len(df[df["Accuracy"] == True])
                row_count =  df['Accuracy'].notna().sum()
                accuracy = 100 * (true_count / row_count)
                
                avg_length = df['Completion'].dropna().str.len().mean()

                print(f"{csv_path}: accuracy: {accuracy:.2f}, mean length: {avg_length:.2f}")
            except Exception as e:
                print(f"{csv_path}: {e}")

    @staticmethod
    def get_filenames():
        dir_ = './integrated_information_theory/inference/causalbench/'
        csv_paths = [
                        'text/run_/settings_0_causalbench.csv',
                        'text/run_/settings_37_causalbench.csv',
                        'text/run_/settings_51_causalbench.csv',
                        'text/run_/settings_46_causalbench.csv',
                        'text/run_/settings_64_causalbench.csv',
                        'text/run_/settings_65_causalbench.csv',
                        
                        # 'math/settings_0_math_causalbench.csv',
                        # 'math/settings_37_math_causalbench.csv',
                        # 'math/settings_51_math_causalbench.csv',
                        # 'math/settings_46_math_causalbench.csv',
                        # 'math/settings_64_math_causalbench.csv',
                        # 'math/settings_65_math_causalbench.csv',

                    ]
        return dir_, csv_paths

    @staticmethod
    def extract_first_number(filename):
        match = re.search(r'\d+\.\d+|\d+', filename)
        if not match:
            return None
        
        return float(match.group()) if '.' in match.group() else int(match.group())

causalbench_accuracy_analysis.calculate_accuracy()
