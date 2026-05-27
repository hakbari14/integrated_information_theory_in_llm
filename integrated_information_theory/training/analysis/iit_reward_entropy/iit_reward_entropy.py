import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc

class iit_reward_entropy(object):

    @staticmethod
    def plot_auroc():
        dir, csv_paths, training_step = iit_reward_entropy.get_filenames()
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        axes = axes.flatten()

        for plot_idx, file_path in enumerate(csv_paths): 
            df = pd.read_csv(f'{dir}/{file_path}')
            iit_reward_entropy.check_columns(df)
            
            df = df[(df["Split"] == "eval") & (df["Trainer_Global_Step"] >= training_step[plot_idx]) & (df["Trainer_Global_Step"] <= training_step[plot_idx])]
            df = df[["Accuracy_Reward", "Entropy", "Completion_Loss", "Phi_Reward", "Phi_Reward_Raw"]].dropna()

            accuracy_list = df['Accuracy_Reward'].tolist()

            min_entropy_val = df['Entropy'].min()
            max_entropy_val = df['Entropy'].max()
            df['confidence_entropy'] = (max_entropy_val - df['Entropy']) / (max_entropy_val - min_entropy_val)
            confidence_entropy_list = df['confidence_entropy'].tolist()

            min_loss_val = df['Completion_Loss'].min()
            max_loss_val = df['Completion_Loss'].max()
            df['confidence_loss'] = (max_entropy_val - df['Completion_Loss']) / (max_loss_val - min_loss_val)
            confidence_loss_list = df['confidence_loss'].tolist()

            iit_reward_list = df['Phi_Reward'].tolist()
            
            y_true = np.array(accuracy_list)
            accuracy = np.average(y_true)
            con_A = np.array(confidence_entropy_list)
            con_B = np.array(confidence_loss_list)
            con_C = np.array(iit_reward_list)

            fpr1, tpr1, _ = roc_curve(y_true, con_A)
            roc_auc1 = auc(fpr1, tpr1)

            fpr2, tpr2, _ = roc_curve(y_true, con_B)
            roc_auc2 = auc(fpr2, tpr2)

            fpr3, tpr3, _ = roc_curve(y_true, con_C)
            roc_auc3 = auc(fpr3, tpr3)
            
            ax = axes[plot_idx]

            ax.plot(fpr1, tpr1, linewidth=2, label=f"Entropy (AUROC = {roc_auc1:.3f})")
            ax.plot(fpr2, tpr2, linewidth=2, label=f"Loss (AUROC = {roc_auc2:.3f})")
            ax.plot(fpr3, tpr3, linewidth=2, label=f"IIT (AUROC = {roc_auc3:.3f})")
            ax.plot([0, 1], [0, 1], [0, 1], linestyle="--", linewidth=1)
            ax.set_title(f"{Path(file_path).stem}, Accuracy= {accuracy:.2f}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()

    
        plt.tight_layout()
        plt.plot()
        plt.savefig(f'integrated_information_theory/training/analysis/iit_reward_entropy/training_auroc.png')

    @staticmethod
    def scatterplot_entropy_iit_reward():
        dir, csv_paths, training_step = iit_reward_entropy.get_filenames()
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        axes = axes.flatten()

        for plot_idx, file_path in enumerate(csv_paths): 
            df = pd.read_csv(f'{dir}/{file_path}')
            
            iit_reward_entropy.check_columns(df)
            df = df[(df["Split"] == "eval") & (df["Trainer_Global_Step"] == training_step[plot_idx])]
            df = df[["Phi_Reward", "Entropy"]].dropna()

            min_entropy_val = df['Entropy'].min()
            max_entropy_val = df['Entropy'].max()
            df['confidence_entropy'] = (max_entropy_val - df['Entropy']) / (max_entropy_val - min_entropy_val)

            # Compute correlation
            correlation = df["Phi_Reward"].corr(df["confidence_entropy"])

            ax = axes[plot_idx]

            ax.scatter(df["confidence_entropy"], df["Phi_Reward"], alpha=0.7, edgecolors='k', color=f'C{plot_idx}')
            ax.set_title(f'{Path(file_path).stem}, PC=({correlation:.3f})')
            ax.set_xlabel("Entropy")
            ax.set_ylabel("IIT Reward")
            ax.grid(True, linestyle='--', alpha=0.6)
    
        plt.tight_layout()
        plt.plot()
        plt.savefig(f'integrated_information_theory/training/analysis/iit_reward_entropy/training_scatterplot.png')

    @staticmethod
    def check_columns(df):
        required_cols = ["Phi_Reward", "Entropy", "Split", "Trainer_Global_Step"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in CSV")
            
        return None    

    @staticmethod
    def get_filenames():
        dir = './integrated_information_theory/training/peft/open_thoughts_deepseekr1_qwen_7/logs/'
        csv_paths = [
                        'settings_46/settings_46.csv',
                        'settings_64/settings_64.csv',
                        'settings_65/settings_65.csv',
                    ]
        training_step = [
                        500,
                        500,
                        1200,
                    ]
        return dir, csv_paths, training_step



# iit_reward_entropy.scatterplot_entropy_iit_reward()
iit_reward_entropy.plot_auroc()
