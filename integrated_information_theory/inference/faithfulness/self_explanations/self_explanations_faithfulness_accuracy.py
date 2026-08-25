import pandas as pd
import re
from collections import defaultdict
import numpy as np 
from integrated_information_theory.utils import my_utils

class self_explanations_faithfulness_accuracy(object):

    @staticmethod
    def calculate_accuracy():
        dir_, csv_paths = self_explanations_faithfulness_accuracy.get_csv_paths()
        grouped_acc, grouped_len = defaultdict(list), defaultdict(list)        
        for csv_path in csv_paths:
            model = self_explanations_faithfulness_accuracy.extract_first_number(csv_path)
            nsg_list = []
            for run in range(1,6):
                try:
                    filepath = f'{dir_}/{csv_path}'.replace('run_', f'run_{run}')
                    df = pd.read_parquet(filepath)
                    nsg = self_explanations_faithfulness_accuracy.calculate_nsg(df)
                    nsg_list.append(nsg)
                   
                except Exception as e:
                    print(f"{csv_path}: {e}")
            
            print(f"{csv_path}: NSG ({np.mean(nsg_list):.3f} - {np.std(nsg_list):.2f})")
            grouped_acc[model].append(np.mean(nsg_list))        

        print()
        for model in grouped_acc:
            print(f'Settings{model} : NSG = {np.mean(grouped_acc[model]):.3f}')            
        

    @staticmethod
    def calculate_nsg(df):
        count_with, count_without = 0,0
        for index, row in df.iterrows():
            reference = df.loc[index, "counterfactual_reference_response_answer"]
            predictor_with = df.loc[index, "counterfactual_predictor_response_with_explanation_answer"]
            predictor_without = df.loc[index, "counterfactual_predictor_response_without_explanation_answer"]
            if predictor_with == reference:
                count_with += 1
            if predictor_without == reference:
                count_without += 1

        acc_with = count_with / len(df)
        acc_without = count_without / len(df)
        return (acc_with - acc_without) / (1 - acc_without)

    @staticmethod
    def get_good_cases(df):
        idx_list = []
        for index, row in df.iterrows():
            reference = df.loc[index, "counterfactual_reference_response_answer"]
            predictor_with = df.loc[index, "counterfactual_predictor_response_with_explanation_answer"]
            predictor_without = df.loc[index, "counterfactual_predictor_response_without_explanation_answer"]
            idx = df.loc[index, "counterfactual_question_idx"]
            if predictor_with == reference and predictor_without != reference:
                idx_list.append(idx)

        return idx_list

    @staticmethod
    def extract_good_case():
        dir_, csv_paths = self_explanations_faithfulness_accuracy.get_csv_paths()
        for csv_path in csv_paths:
            try:
                df = pd.read_parquet(dir_+csv_path)
                idx_list = self_explanations_faithfulness_accuracy.get_good_cases(df)
                print(f" ----------------------  {csv_path} ----------------------")
                print(idx_list)
            except Exception as e:
                print(f"{csv_path}: {e}")

    @staticmethod
    def print(idx):
        dir_, csv_paths = self_explanations_faithfulness_accuracy.get_csv_paths()
        for csv_path in csv_paths:
            try:
                filepath = f'{dir_}/{csv_path}'.replace('run_', f'run_{1}')
                df = pd.read_parquet(filepath)
                for index, row in df.iterrows():
                    if df.loc[index, "counterfactual_question_idx"] != idx: continue
                    print(f" ----------------------  {csv_path} ----------------------")
                    print(f'counterfactual_question_prompt')
                    print(df.loc[index, "counterfactual_question_prompt"])
                    print(f'Ansert = {df.loc[index, "counterfactual_reference_response_answer"]}')

                    print(f" ---------------------------------------------")

                    print(f'counterfactual_predictor_response_with_explanation_raw_response')
                    print(df.loc[index, "counterfactual_predictor_response_with_explanation_raw_response"])
                    print(f'Ansert = {df.loc[index, "counterfactual_predictor_response_with_explanation_answer"]}')
                    print(f" ---------------------------------------------")

                    print(f'counterfactual_predictor_response_without_explanation_raw_response')
                    print(df.loc[index, "counterfactual_predictor_response_without_explanation_raw_response"])
                    print(f'Ansert = {df.loc[index, "counterfactual_predictor_response_without_explanation_answer"]}')

            except Exception as e:
                print(f"{csv_path}: {e}")

    @staticmethod
    def get_csv_paths():
        dir_ = 'integrated_information_theory/inference/faithfulness/self_explanations/'
        csv_paths = [
                        
                        # 'settings_0/run_/predictor_answers_settings_0.parquet',
                        # 'settings_37/run_/predictor_answers_settings_37.parquet',
                        # 'settings_51/run_/predictor_answers_settings_51.parquet',
                        'settings_46/run_/predictor_answers_settings_46.parquet',
                        # 'settings_64/run_/predictor_answers_settings_64.parquet',
                        # 'settings_65/run_/predictor_answers_settings_65.parquet',
                        'settings_83/run_/predictor_answers_settings_83.parquet',
                    ]
        return dir_, csv_paths

    @staticmethod
    def extract_first_number(filename):
        match = re.search(r'\d+\.\d+|\d+', filename)
        if not match:
            return None
        
        return float(match.group()) if '.' in match.group() else int(match.group())

    @staticmethod
    def convert_csv():
        dir_, csv_paths = self_explanations_faithfulness_accuracy.get_csv_paths()
        for csv_path in csv_paths:
            filepath = f'{dir_}/{csv_path}'.replace('run_', f'run_{1}')
            my_utils.convert_parquet_to_csv(filepath)


# self_explanations_faithfulness_accuracy.calculate_accuracy()
# self_explanations_faithfulness_accuracy.extract_good_case()
# self_explanations_faithfulness_accuracy.print(758)
self_explanations_faithfulness_accuracy.calculate_accuracy()


