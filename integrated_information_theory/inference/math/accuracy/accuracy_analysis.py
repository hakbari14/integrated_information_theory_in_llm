import pandas as pd
import re
import numpy as np 
from integrated_information_theory.datasets.math.gpqa_dataset import gpqa_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config

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
    def update_final_answer():
        for i in range(1,6):
            log_file_name = './integrated_information_theory/inference/math/accuracy/settings_79/run_/settings_79_gpqa_full.csv'
            log_file_name = log_file_name.replace('run_', f'run_{i}')
            df = pd.read_csv(log_file_name)
            config = dataset_config('/home/hr_akbari/research/integrated_information_theory_in_llm/live_logs/settings_79/checkpoint-1150-HF')
            dataset = gpqa_dataset(config)

            for index, row in df.iterrows():
                if not pd.isna(df.loc[index, "Final_Answer"]): continue
                if len(df.loc[index, "Completion"]) == 0: continue

                prompt = df.loc[index, "Prompt"]
                completion = df.loc[index, "Completion"]
                target = df.loc[index, "Target"]
                final_answer, target_answer_equal, _ = dataset.extract_and_verify_final_answer(prompt, completion , target)
                df.at[index, "Final_Answer"] = final_answer
                df.at[index, "Accuracy"] = target_answer_equal

            print('*****************************************************************************')
            df.to_csv(log_file_name, index=False)            

    @staticmethod
    def get_csv_paths():
        dir = './integrated_information_theory/inference'
        csv_paths = {
            "open-thoughts_settings_0": {
                        "file_paths": "math/accuracy/settings_0/run_/settings_0_open_thoughts_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 2,
                        },
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
            "aime_settings_80": {
                        "file_paths": "math/accuracy/settings_80/run_/settings_80_aime_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "open-thoughts_settings_00": {
                        "file_paths": "math/accuracy/settings_00/run_/settings_00_open_thoughts_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 2,
                        },
            "aime_settings_00": {
                        "file_paths": "math/accuracy/settings_00/run_/settings_00_aime_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "aime_settings_81": {
                        "file_paths": "math/accuracy/settings_81/run_/settings_81_aime_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "aime_settings_82": {
                        "file_paths": "math/accuracy/settings_82/run_/settings_82_aime_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "aime_settings_79": {
                        "file_paths": "math/accuracy/settings_79/run_/settings_79_aime_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "aime_settings_83": {
                        "file_paths": "math/accuracy/settings_83/run_/settings_83_aime_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "aime_settings_84": {
                        "file_paths": "math/accuracy/settings_84/run_/settings_84_aime_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
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
            "math500_settings_00": {
                        "file_paths": "math/accuracy/settings_00/run_/settings_00_math500_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "math500_settings_81": {
                        "file_paths": "math/accuracy/settings_81/run_/settings_81_math500_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "math500_settings_82": {
                        "file_paths": "math/accuracy/settings_82/run_/settings_82_math500_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "math500_settings_79": {
                        "file_paths": "math/accuracy/settings_79/run_/settings_79_math500_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "math500_settings_83": {
                        "file_paths": "math/accuracy/settings_83/run_/settings_83_math500_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "math500_settings_84": {
                        "file_paths": "math/accuracy/settings_84/run_/settings_84_math500_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
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
            "gsm8k_settings_00": {
                        "file_paths": "math/accuracy/settings_00/run_/settings_00_gsm8k_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "gsm8k_settings_81": {
                        "file_paths": "math/accuracy/settings_81/run_/settings_81_gsm8k_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "gsm8k_settings_82": {
                        "file_paths": "math/accuracy/settings_82/run_/settings_82_gsm8k_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "gsm8k_settings_79": {
                        "file_paths": "math/accuracy/settings_79/run_/settings_79_gsm8k_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "gsm8k_settings_83": {
                        "file_paths": "math/accuracy/settings_83/run_/settings_83_gsm8k_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "gsm8k_settings_84": {
                        "file_paths": "math/accuracy/settings_84/run_/settings_84_gsm8k_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
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
            "gpqa_settings_00": {
                        "file_paths": "math/accuracy/settings_00/run_/settings_00_gpqa_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "gpqa_settings_81": {
                        "file_paths": "math/accuracy/settings_81/run_/settings_81_gpqa_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "gpqa_settings_82": {
                        "file_paths": "math/accuracy/settings_82/run_/settings_82_gpqa_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "gpqa_settings_79": {
                        "file_paths": "math/accuracy/settings_79/run_/settings_79_gpqa_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "gpqa_settings_83": {
                        "file_paths": "math/accuracy/settings_83/run_/settings_83_gpqa_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "gpqa_settings_84": {
                        "file_paths": "math/accuracy/settings_84/run_/settings_84_gpqa_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
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
            "countdown_settings_00": {
                        "file_paths": "math/accuracy/settings_00/run_/settings_00_countdown_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "countdown_settings_81": {
                        "file_paths": "math/accuracy/settings_81/run_/settings_81_countdown_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "countdown_settings_82": {
                        "file_paths": "math/accuracy/settings_82/run_/settings_82_countdown_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "countdown_settings_79": {
                        "file_paths": "math/accuracy/settings_79/run_/settings_79_countdown_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "countdown_settings_83": {
                        "file_paths": "math/accuracy/settings_83/run_/settings_83_countdown_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
                        },
            "countdown_settings_84": {
                        "file_paths": "math/accuracy/settings_84/run_/settings_84_countdown_full.csv",
                        "from_run_number": 1,
                        "to_run_number": 6,
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
        
        return float(match.group()) if '.' in match.group() else match.group()


accuracy_analysis.calculate_accuracy()
# accuracy_analysis.update_final_answer()
