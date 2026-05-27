import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import re
from sklearn.metrics import roc_curve, auc

class plot_confidence(object):

    @staticmethod
    def plot_auroc(inference_name, y_true, confidence_A):
        y_true = np.array(y_true)
        con_A = np.array(confidence_A)
        
        fpr1, tpr1, _ = roc_curve(y_true, con_A)
        roc_auc1 = auc(fpr1, tpr1)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr1, tpr1, linewidth=2, label=f"Metric 1 (AUROC = {roc_auc1:.3f})")
        plt.plot([0, 1], linestyle="--", linewidth=1)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for LLM Confidence Metrics")
        plt.legend()
        plt.grid(True)
        plt.plot()
        plt.savefig(f'integrated_information_theory/inference/analysis/confidence/{inference_name}_auroc.png')

    @staticmethod
    def plot(run_number = 1):
        dir, datasets, csv_paths = plot_confidence.get_filenames()
        for dataset in datasets: 
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            axes = axes.flatten()

            for plot_idx, file_path in enumerate(csv_paths): 
                if dataset not in file_path: continue
                
                settings = plot_confidence.extract_first_number(file_path)
                file_path = file_path.replace('run_', f'run_{run_number}')
                df = pd.read_csv(f'{dir}/{file_path}')
                plot_confidence.check_columns(df)
                
                df = df[["Accuracy", "Entropy", "Completion_Loss"]].dropna()
                df['Accuracy_Reward'] = df['Accuracy'].map({True: 1, False: 0})        
                accuracy_list = df['Accuracy_Reward'].tolist()

                min_entropy_val = df['Entropy'].min()
                max_entropy_val = df['Entropy'].max()
                df['confidence_entropy'] = (max_entropy_val - df['Entropy']) / (max_entropy_val - min_entropy_val)
                confidence_entropy_list = df['confidence_entropy'].tolist()

                min_loss_val = df['Completion_Loss'].min()
                max_loss_val = df['Completion_Loss'].max()
                df['confidence_loss'] = (max_entropy_val - df['Completion_Loss']) / (max_loss_val - min_loss_val)
                confidence_loss_list = df['confidence_loss'].tolist()
                
                y_true = np.array(accuracy_list)
                con_A = np.array(confidence_entropy_list)
                con_B = np.array(confidence_loss_list)

                fpr1, tpr1, _ = roc_curve(y_true, con_A)
                roc_auc1 = auc(fpr1, tpr1)

                fpr2, tpr2, _ = roc_curve(y_true, con_B)
                roc_auc2 = auc(fpr2, tpr2)
                
                ax = axes[plot_idx]

                ax.plot(fpr1, tpr1, linewidth=2, label=f"Entropy (AUROC = {roc_auc1:.3f})")
                ax.plot(fpr2, tpr2, linewidth=2, label=f"Loss (AUROC = {roc_auc2:.3f})")
                ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
                ax.set_title(f"Settings_{settings}")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend()

        
            plt.tight_layout()
            plt.plot()
            plt.savefig(f'integrated_information_theory/inference/analysis/confidence/{dataset}_auroc.png')

    @staticmethod
    def check_columns(df):
        required_cols = ["Accuracy", "Entropy", "Completion_Loss"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in CSV")
            
        return None    

    @staticmethod
    def get_filenames():
        dir = './integrated_information_theory/inference/math/accuracy'
        csv_paths = [
                        'settings_0/run_/settings_0_aime_full.csv',
                        'settings_37/run_/settings_37_aime_full.csv',
                        'settings_51/run_/settings_51_aime_full.csv',
                        'settings_46/run_/settings_46_aime_full.csv',
                        'settings_64/run_/settings_64_aime_full.csv',
                        'settings_65/run_/settings_65_aime_full.csv',

                    ]

        datasets = [
                        'aime',
                        # 'countdown',
                        # 'gpqa',
                        # 'gsm8k',
                        # 'math500',
                    ]
        return dir, datasets, csv_paths

    @staticmethod
    def extract_first_number(filename):
        match = re.search(r'\d+\.\d+|\d+', filename)
        if not match:
            return None
        
        return float(match.group()) if '.' in match.group() else int(match.group())

plot_confidence.plot()
