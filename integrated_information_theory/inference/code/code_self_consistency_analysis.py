import pandas as pd

class self_consistency_analysis(object):

    @staticmethod
    def calculate_accuracy():
        dir_ = 'integrated_information_theory/inference/code/'
        csv_paths = [
                        'humaneval/self_consistency/settings_0_humaneval_sc.csv', 
                        'humaneval/self_consistency/settings_37_humaneval_sc.csv', 
                        'humaneval/self_consistency/settings_51_humaneval_sc.csv', 
                        'humaneval/self_consistency/settings_46_humaneval_sc.csv', 
                        'humaneval/self_consistency/settings_64_humaneval_sc.csv', 
                        'humaneval/self_consistency/settings_65_humaneval_sc.csv', 
                    ]

        for csv_path in csv_paths:
            try:
                df = pd.read_csv(dir_+csv_path)
                true_count = len(df[df["Accuracy"] == True])
                row_count = len(df["Accuracy"])
                accuracy = 100 * (true_count / row_count)


                pass_at_k = df["Pass_at_k"].sum()
                pass_at_k = 100 * (pass_at_k / row_count)

                print(f"{csv_path}: accuracy = {accuracy:.2f}, pass_@_k = {pass_at_k:.2f}")
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


self_consistency_analysis.calculate_accuracy()
# self_consistency_analysis.groupy_by()
