import pandas as pd
import numpy as np 

class code_accuracy_analysis(object):

    @staticmethod
    def calculate_accuracy():
        dir_, csv_paths = code_accuracy_analysis.get_csv_paths()
        for csv_path in csv_paths:
            acc_list, mean_list = [], []
            for run in range(6,7):
                try:
                    df = pd.read_csv(f'{dir_}/run{run}/{csv_path}')

                    true_count = len(df[df["Accuracy"] == True])
                    row_count = len(df["Accuracy"])
                    accuracy = 100 * (true_count / row_count)
                    acc_list.append(accuracy)

                    token_count = sum(df["Token_Count"].to_list())
                    mean_token = token_count / row_count
                    mean_list.append(mean_token)
                except Exception as e:
                    print(f"{csv_path}: {e}")
            
            print(f"{csv_path}: Accuracy({np.mean(acc_list):.2f} - {np.std(acc_list):.2f}), Response Length({np.mean(mean_list):.2f} - {np.std(mean_list):.2f})")
            

    @staticmethod
    def get_csv_paths():
        dir_ = './integrated_information_theory/inference/code/humaneval/accuracy/'
        csv_paths = [
                        'settings_0_humaneval.csv',
                        # 'settings_37_humaneval.csv',
                        # 'settings_51_humaneval.csv',
                        # 'settings_46_humaneval.csv',
                        # 'settings_64_humaneval.csv',
                        # 'settings_65_humaneval.csv',
                    ]
        return dir_, csv_paths



code_accuracy_analysis.calculate_accuracy()
