import pandas as pd
import re 
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class consciousness_accuracy_analysis(object):

    @staticmethod
    def calculate_accuracy():
        dir_, csv_paths = consciousness_accuracy_analysis.get_filenames()
        grouped_acc, grouped_len = defaultdict(list), defaultdict(list)        
        for csv_path in csv_paths:
            model = consciousness_accuracy_analysis.extract_first_number(csv_path)
            acc_list, mean_list = [], []
            for run in range(1,6):
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
            
            print(f"{csv_path}: Accuracy({np.mean(acc_list):.2f} - {np.std(acc_list):.2f}), Response Length({np.mean(mean_list):.0f} - {np.std(mean_list):.2f})")
            grouped_acc[model].append(np.mean(acc_list))        
            grouped_len[model].append(np.mean(mean_list))        

        print()
        for model in grouped_acc:
            print(f'Settings{model} : Accuracy = {np.mean(grouped_acc[model]):.2f}, Mean Token Length = {np.mean(grouped_len[model]):.0f}')            
        

    @staticmethod
    def create_radar_chart():
        dir_ = './integrated_information_theory/inference/consciousness/'
        for settings in ['0', '37', '51', '46', '64', '65']:
            accuracy_metric = []
            for bm in consciousness_accuracy_analysis.get_benchmark_name():
                df = pd.read_csv(f'{dir_}/{bm}/settings_{settings}_{bm}.csv')

                true_count = len(df[df["Accuracy"] == True])
                row_count = len(df["Accuracy"])
                accuracy = 100 * (true_count / row_count)
                accuracy_metric.append(accuracy)


            benchmarks = consciousness_accuracy_analysis.get_UI_benchmark_name()        
            N = len(benchmarks)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

            accuracy_metric += accuracy_metric[:1]
            angles += angles[:1]

            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, polar=True)

            ax.plot(angles, accuracy_metric, linewidth=2)
            ax.fill(angles, accuracy_metric, alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(benchmarks)

            plt.title("Consciousness Evaluation")
            plt.plot()
            plt.savefig(f'./integrated_information_theory/inference/consciousness/consciousness_evaluation_settings_{settings}.png')
        
        return None

    @staticmethod
    def refine_final_answer():
        dir_, csv_paths = consciousness_accuracy_analysis.get_filenames()
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(dir_+csv_path)
                for index, row in df.iterrows():
                    if df.loc[index, "Accuracy"] == False and pd.isna(df.loc[index, "Final_Answer"]):
                        final_answer = consciousness_accuracy_analysis.final_answer_extraction(df.loc[index, "Completion"])
                        df.at[index, "Final_Answer"] = final_answer
                        df.at[index, "Accuracy"] = final_answer == df.loc[index, "Target"]

                df.to_csv(dir_+csv_path, index=False)            
            except Exception as e:
                print(f"{csv_path}: {e}")

    @staticmethod
    def final_answer_extraction(solution):
        last = solution[-min(600, len(solution)):]


        patterns = [
            r'(?i)Answer.*?([AB])\b',            
            r'(?i)boxed\s*\{\s*([ABab])\s*\}',
        ]

        for pattern in patterns:
            match = re.search(pattern, last, re.IGNORECASE | re.DOTALL)
            if not match: continue
            answer = match.group(1).upper()
            if answer not in ['A', 'B']: continue
            return answer
        
        return None

    @staticmethod
    def get_filenames():
        dir_ = './integrated_information_theory/inference/consciousness'
        csv_paths = [
                        'sequential_planning/run_/settings_0_sequential_planning.csv',
                        'sequential_planning/run_/settings_37_sequential_planning.csv',
                        'sequential_planning/run_/settings_51_sequential_planning.csv',
                        'sequential_planning/run_/settings_46_sequential_planning.csv',
                        'sequential_planning/run_/settings_64_sequential_planning.csv',
                        'sequential_planning/run_/settings_65_sequential_planning.csv',

                        'self_improve/run_/settings_0_self_improve.csv',
                        'self_improve/run_/settings_37_self_improve.csv',
                        'self_improve/run_/settings_51_self_improve.csv',
                        'self_improve/run_/settings_46_self_improve.csv',
                        'self_improve/run_/settings_64_self_improve.csv',
                        'self_improve/run_/settings_65_self_improve.csv',

                        'self_reflection/run_/settings_0_self_reflection.csv',
                        'self_reflection/run_/settings_37_self_reflection.csv',
                        'self_reflection/run_/settings_51_self_reflection.csv',
                        'self_reflection/run_/settings_46_self_reflection.csv',
                        'self_reflection/run_/settings_64_self_reflection.csv',
                        'self_reflection/run_/settings_65_self_reflection.csv',

                        'known_unknowns/run_/settings_0_known_unknowns.csv',
                        'known_unknowns/run_/settings_37_known_unknowns.csv',
                        'known_unknowns/run_/settings_51_known_unknowns.csv',
                        'known_unknowns/run_/settings_46_known_unknowns.csv',
                        'known_unknowns/run_/settings_64_known_unknowns.csv',
                        'known_unknowns/run_/settings_65_known_unknowns.csv',

                        'known_knowns/run_/settings_0_known_knowns.csv',
                        'known_knowns/run_/settings_37_known_knowns.csv',
                        'known_knowns/run_/settings_51_known_knowns.csv',
                        'known_knowns/run_/settings_46_known_knowns.csv',
                        'known_knowns/run_/settings_64_known_knowns.csv',
                        'known_knowns/run_/settings_65_known_knowns.csv',

                        'situational_awareness/run_/settings_0_situational_awareness.csv',
                        'situational_awareness/run_/settings_37_situational_awareness.csv',
                        'situational_awareness/run_/settings_51_situational_awareness.csv',
                        'situational_awareness/run_/settings_46_situational_awareness.csv',
                        'situational_awareness/run_/settings_64_situational_awareness.csv',
                        'situational_awareness/run_/settings_65_situational_awareness.csv',

                        'intention/run_/settings_0_intention.csv',
                        'intention/run_/settings_37_intention.csv',
                        'intention/run_/settings_51_intention.csv',
                        'intention/run_/settings_46_intention.csv',
                        'intention/run_/settings_64_intention.csv',
                        'intention/run_/settings_65_intention.csv',

                        'deception/run_/settings_0_deception.csv',
                        'deception/run_/settings_37_deception.csv',
                        'deception/run_/settings_51_deception.csv',
                        'deception/run_/settings_46_deception.csv',
                        'deception/run_/settings_64_deception.csv',
                        'deception/run_/settings_65_deception.csv',

                        'harm/run_/settings_0_harm.csv',
                        'harm/run_/settings_37_harm.csv',
                        'harm/run_/settings_51_harm.csv',
                        'harm/run_/settings_46_harm.csv',
                        'harm/run_/settings_64_harm.csv',
                        'harm/run_/settings_65_harm.csv',

                        'belief/run_/settings_0_belief.csv',
                        'belief/run_/settings_37_belief.csv',
                        'belief/run_/settings_51_belief.csv',
                        'belief/run_/settings_46_belief.csv',
                        'belief/run_/settings_64_belief.csv',
                        'belief/run_/settings_65_belief.csv',
                    ]
        return dir_, csv_paths

    @staticmethod
    def get_benchmark_name():
        return ['sequential_planning', 'self_improve', 'self_reflection', 'known_unknowns', 'known_knowns', 'situational_awareness', 'intention', 'deception', 'harm', 'belief']

    @staticmethod
    def get_UI_benchmark_name():
        ui_list = []
        for b in consciousness_accuracy_analysis.get_benchmark_name():
            ui_list.append(consciousness_accuracy_analysis.capitalize_after_underscore(b))
        return ui_list
    
    @staticmethod
    def capitalize_after_underscore(text):
        parts = text.split('_')
        capitalized_parts = [part[0].upper() + part[1:] if part else '' for part in parts]
        return ' '.join(capitalized_parts)

    @staticmethod
    def extract_first_number(filename):
        match = re.search(r'\d+\.\d+|\d+', filename)
        if not match:
            return None
        
        return float(match.group()) if '.' in match.group() else int(match.group())

consciousness_accuracy_analysis.calculate_accuracy()
# consciousness_accuracy_analysis.create_radar_chart()
