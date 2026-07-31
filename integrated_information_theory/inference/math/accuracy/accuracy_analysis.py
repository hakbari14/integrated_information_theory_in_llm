import pandas as pd
import re
import numpy as np 

class accuracy_analysis(object):

    @staticmethod
    def calculate_accuracy():
        dir, csv_paths = accuracy_analysis.get_csv_paths()
        data_list = []
        for dataset_settings, csv_dataset in csv_paths.items():
            csv_path = csv_dataset['file_paths']
            from_run_number = csv_dataset['from_run_number']
            to_run_number = csv_dataset['to_run_number']
            model = accuracy_analysis.extract_first_number(csv_path)
            dataset = dataset_settings.split("_", 1)[0]
            
            for run_number in range(from_run_number,to_run_number):
                try:
                    filepath = f'{dir}/{csv_path}'.replace('run_', f'run_{run_number}')
                    df = pd.read_csv(filepath)

                    true_count = len(df[df["Accuracy"] == True])
                    row_count = len(df["Accuracy"])
                    accuracy = 100 * (true_count / row_count)

                    token_count = sum(df["Token_Count"].to_list())
                    avg_length = token_count / row_count
                    
                    data_item = {
                                    "run_number": run_number, 
                                    "model": model , 
                                    "dataset": dataset , 
                                    "accuracy": accuracy,
                                    "mean_length": avg_length,
                                }
                    data_list.append(data_item)
                    
                except Exception as e:
                    print(f"{csv_path}: {e}")

        df_summary = pd.DataFrame(data_list)
        group_cols=['model']        
        value_cols=['accuracy', 'mean_length']
        df_summary_model = accuracy_analysis.aggregate_mean_pandas_rounded(df_summary, group_cols, value_cols)
        df_summary_model = df_summary_model.sort_values(by=['model'])        
        print(df_summary_model.to_string(index=False))        
        print()

        group_cols=['model', 'dataset']        
        value_cols=['accuracy', 'mean_length']
        df_summary_model_dataset = accuracy_analysis.aggregate_mean_pandas_rounded(df_summary, group_cols, value_cols)
        df_summary_model_dataset = df_summary_model_dataset.sort_values(by=['dataset', 'model'])        
        print(df_summary_model_dataset.to_string(index=False))        


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
        dir = './integrated_information_theory/inference'
        csv_paths = {
            "aime_settings_0": {
                        "file_paths": "math/accuracy/settings_0/run_/settings_0_aime_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "aime_settings_37": {
                        "file_paths": "math/accuracy/settings_37/run_/settings_37_aime_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "aime_settings_51": {
                        "file_paths": "math/accuracy/settings_51/run_/settings_51_aime_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "aime_settings_46": {
                        "file_paths": "math/accuracy/settings_46/run_/settings_46_aime_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "aime_settings_64": {
                        "file_paths": "math/accuracy/settings_64/run_/settings_64_aime_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "aime_settings_65": {
                        "file_paths": "math/accuracy/settings_65/run_/settings_65_aime_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },

            "math500_settings_0": {
                        "file_paths": "math/accuracy/settings_0/run_/settings_0_math500_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "math500_settings_37": {
                        "file_paths": "math/accuracy/settings_37/run_/settings_37_math500_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "math500_settings_51": {
                        "file_paths": "math/accuracy/settings_51/run_/settings_51_math500_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "math500_settings_46": {
                        "file_paths": "math/accuracy/settings_46/run_/settings_46_math500_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "math500_settings_64": {
                        "file_paths": "math/accuracy/settings_64/run_/settings_64_math500_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "math500_settings_65": {
                        "file_paths": "math/accuracy/settings_65/run_/settings_65_math500_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },

            "gsm8k_settings_0": {
                        "file_paths": "math/accuracy/settings_0/run_/settings_0_gsm8k_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "gsm8k_settings_37": {
                        "file_paths": "math/accuracy/settings_37/run_/settings_37_gsm8k_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "gsm8k_settings_51": {
                        "file_paths": "math/accuracy/settings_51/run_/settings_51_gsm8k_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "gsm8k_settings_46": {
                        "file_paths": "math/accuracy/settings_46/run_/settings_46_gsm8k_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "gsm8k_settings_64": {
                        "file_paths": "math/accuracy/settings_64/run_/settings_64_gsm8k_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "gsm8k_settings_65": {
                        "file_paths": "math/accuracy/settings_65/run_/settings_65_gsm8k_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },


            "gpqa_settings_0": {
                        "file_paths": "math/accuracy/settings_0/run_/settings_0_gpqa_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "gpqa_settings_37": {
                        "file_paths": "math/accuracy/settings_37/run_/settings_37_gpqa_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "gpqa_settings_51": {
                        "file_paths": "math/accuracy/settings_51/run_/settings_51_gpqa_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "gpqa_settings_46": {
                        "file_paths": "math/accuracy/settings_46/run_/settings_46_gpqa_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "gpqa_settings_64": {
                        "file_paths": "math/accuracy/settings_64/run_/settings_64_gpqa_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "gpqa_settings_65": {
                        "file_paths": "math/accuracy/settings_65/run_/settings_65_gpqa_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },


            "countdown_settings_0": {
                        "file_paths": "math/accuracy/settings_0/run_/settings_0_countdown_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "countdown_settings_37": {
                        "file_paths": "math/accuracy/settings_37/run_/settings_37_countdown_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "countdown_settings_51": {
                        "file_paths": "math/accuracy/settings_51/run_/settings_51_countdown_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "countdown_settings_46": {
                        "file_paths": "math/accuracy/settings_46/run_/settings_46_countdown_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "countdown_settings_64": {
                        "file_paths": "math/accuracy/settings_64/run_/settings_64_countdown_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },
            "countdown_settings_65": {
                        "file_paths": "math/accuracy/settings_65/run_/settings_65_countdown_full.csv",
                        "from_run_number": 11,
                        "to_run_number": 16,
                        },


            "humaneval_settings_0": {
                        "file_paths": "code/humaneval/accuracy/run_/settings_0_humaneval.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "humaneval_settings_37": {
                        "file_paths": "code/humaneval/accuracy/run_/settings_37_humaneval.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "humaneval_settings_51": {
                        "file_paths": "code/humaneval/accuracy/run_/settings_51_humaneval.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "humaneval_settings_46": {
                        "file_paths": "code/humaneval/accuracy/run_/settings_46_humaneval.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "humaneval_settings_64": {
                        "file_paths": "code/humaneval/accuracy/run_/settings_64_humaneval.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "humaneval_settings_65": {
                        "file_paths": "code/humaneval/accuracy/run_/settings_65_humaneval.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },

        }
        
        return dir, csv_paths


    @staticmethod
    def aggregate_mean_pandas_rounded(df, group_cols, value_cols) -> pd.DataFrame:
        result = df.groupby(group_cols)[value_cols].mean().reset_index()
        for col in value_cols:
            result[col] = result[col].round(3)
        return result

    @staticmethod
    def extract_first_number(filename):
        match = re.search(r'\d+\.\d+|\d+', filename)
        if not match:
            return None
        
        return float(match.group()) if '.' in match.group() else int(match.group())


accuracy_analysis.calculate_accuracy()
# accuracy_analysis.find_mean_length()
