import pandas as pd

class self_explanations_faithfulness_accuracy(object):

    @staticmethod
    def calculate_accuracy():
        dir_, csv_paths = self_explanations_faithfulness_accuracy.get_csv_paths()
        for csv_path in csv_paths:
            try:
                df = pd.read_parquet(dir_+csv_path)
                nsg = self_explanations_faithfulness_accuracy.calculate_nsg(df)
                print(f"{csv_path}: NSG: {nsg:.2f}")
            except Exception as e:
                print(f"{csv_path}: {e}")

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

        acc_with = 100 * (count_with / len(df))
        acc_without = 100 * (count_without / len(df))
        return round(100 * (acc_with - acc_without) / (100 - acc_without), 2)

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
                df = pd.read_parquet(dir_+csv_path)
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
        dir_ = '/home/hr_akbari/research/faithfulness/experiments/'
        csv_paths = [
                        
                        'predictor_answers_settings_0.parquet',
                        # 'predictor_answers_settings_37.parquet',
                        # 'predictor_answers_settings_51.parquet',
                        'predictor_answers_settings_46.parquet',
                        # 'predictor_answers_settings_64.parquet',
                        # 'predictor_answers_settings_65.parquet',
                    ]
        return dir_, csv_paths



# self_explanations_faithfulness_accuracy.calculate_accuracy()
# self_explanations_faithfulness_accuracy.extract_good_case()
self_explanations_faithfulness_accuracy.print(758)


