import pandas as pd
import math

class confidence_calibration(object):
    
    @staticmethod
    def calculate(csv_filename, n_bins = 10):
        df = pd.read_csv(csv_filename)
        
        df['accuracy_reward'] = df['Accuracy'].map({True: 1, False: 0})        

        min_entropy_val = df['Entropy'].min()
        max_entropy_val = df['Entropy'].max()
        df['confidence'] = (max_entropy_val - df['Entropy']) / (max_entropy_val - min_entropy_val)
        
        df['binned_confidence'] = pd.qcut(df['confidence'], q=n_bins)
        agg_perplexity = df.groupby('binned_confidence')['confidence'].agg(['mean'])
        agg_accuracy = df.groupby('binned_confidence')['accuracy_reward'].agg(['mean'])

        expected_calibration_error = 0
        maximum_calibration_error = 0
        for idx, row in enumerate(agg_perplexity.iterrows()):
            confidence = row[1]['mean']
            accuracy = agg_accuracy.iloc[idx]['mean']
            expected_calibration_error += abs(confidence - accuracy)
            maximum_calibration_error = max(abs(confidence - accuracy), maximum_calibration_error)

        expected_calibration_error = expected_calibration_error / (idx + 1)
        return expected_calibration_error, maximum_calibration_error


dir_ = '/home/hr_akbari/research/LLM_PostTraining/integrated_information_theory/inference/math/accuracy/'
csv_paths = [
                # 'settings_0/settings_0_math500_full.csv',
                # 'settings_37/settings_37_math500_full.csv',
                # 'settings_51/settings_51_math500_full.csv',
                # 'settings_46/settings_46_math500_full.csv',
                # 'settings_64/settings_64_math500_full.csv',
                # 'settings_65/settings_65_math500_full.csv',
                
                # 'settings_0/settings_0_gsm8k_full.csv',
                # 'settings_37/settings_37_gsm8k_full.csv',
                # 'settings_51/settings_51_gsm8k_full.csv',
                # 'settings_46/settings_46_gsm8k_full.csv',
                # 'settings_64/settings_64_gsm8k_full.csv',
                # 'settings_65/settings_65_gsm8k_full.csv',
                
                # 'settings_0/settings_0_gpqa_full.csv',
                # 'settings_37/settings_37_gpqa_full.csv',
                # 'settings_51/settings_51_gpqa_full.csv',
                # 'settings_46/settings_46_gpqa_full.csv',
                # 'settings_64/settings_64_gpqa_full.csv',
                # 'settings_65/settings_65_gpqa_full.csv',

                # 'settings_0/settings_0_countdown_full.csv',
                # 'settings_37/settings_37_countdown_full.csv',
                # 'settings_51/settings_51_countdown_full.csv',
                # 'settings_46/settings_46_countdown_full.csv',
                # 'settings_64/settings_64_countdown_full.csv',
                # 'settings_65/settings_65_countdown_full.csv',

                'settings_0/run_1/settings_0_aime_full.csv',
                # 'settings_37/settings_37_aime_full.csv',
                # 'settings_51/settings_51_aime_full.csv',
                'settings_46/run_1/settings_46_aime_full.csv',
                'settings_64/run_1/settings_64_aime_full.csv',
                'settings_65/run_1/settings_65_aime_full.csv',
            ]

for csv_path in csv_paths:
    try:
        ece, mce = confidence_calibration.calculate(dir_+csv_path)
        print(f'{csv_path} : ECE ={ece:.2f}, MCE ={mce:.2f}')
    except Exception as e:
        print(f"{csv_path} : {e}")

