import pandas as pd

class faithfulness_accuracy_analysis(object):

    @staticmethod
    def calculate_accuracy():
        dir_, csv_paths = faithfulness_accuracy_analysis.get_filenames()
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(dir_+csv_path)

                true_count = len(df[df["Accuracy"] == True])
                row_count =  df['Accuracy'].notna().sum()
                accuracy = 100 * (true_count / row_count)
                
                print(f"{csv_path}: accuracy: {accuracy:.2f}")
            except Exception as e:
                print(f"{csv_path}: {e}")

    @staticmethod
    def get_filenames():
        dir_ = './integrated_information_theory/inference/faithfulness/'
        csv_paths = [
                        'post_hoc_rationalization/settings_0_faithfulness_full.csv',
                        'post_hoc_rationalization/settings_37_faithfulness_full.csv',
                        'post_hoc_rationalization/settings_51_faithfulness_full.csv',
                        'post_hoc_rationalization/settings_46_faithfulness_full.csv',
                        'post_hoc_rationalization/settings_49_faithfulness_full.csv',
                        'post_hoc_rationalization/settings_64_faithfulness_full.csv',
                        'post_hoc_rationalization/settings_65_faithfulness_full.csv',

                    ]
        return dir_, csv_paths

faithfulness_accuracy_analysis.calculate_accuracy()
