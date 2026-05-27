import pandas as pd
import matplotlib.pyplot as plt

class iit_reward_completion_length(object):

    @staticmethod
    def plot_mean_length_iit_reward(model_name, last_training_step):
        dir = 'integrated_information_theory/training/peft/open_thoughts_deepseekr1_qwen_7/logs'
        df = pd.read_csv(f'{dir}/{model_name}/{model_name}.csv')

        # Check required columns
        required_cols = ["Phi_Reward_Raw", "Completion", "Split", "Trainer_Global_Step"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in CSV")

        # Filter rows: split == 'eval' and step
        df = df[(df["Split"] == "eval") & (df["Trainer_Global_Step"] == last_training_step)]

        # Drop missing values
        df = df[["Phi_Reward_Raw", "Completion"]].dropna()

        # Compute completion length
        df["completion_length"] = df["Completion"].astype(str).apply(len)

        # Compute correlation
        correlation = df["Phi_Reward_Raw"].corr(df["completion_length"])

        # Scatter plot
        plt.figure()
        plt.scatter(df["completion_length"], df["Phi_Reward_Raw"])
        plt.xlabel("Completion Length")
        plt.ylabel("IIT Reward")
        # plt.title(f"Phi Reward vs Completion Length (eval, step={last_training_step})")
        plt.title(f"Pearson correlation: {round(correlation,3)}")
        plt.plot()
        plt.savefig(f'./integrated_information_theory/training/analysis/iit_reward_completion_length/{model_name}_scatterplot.png')

        print(f"Pearson correlation: {round(correlation,2)}")        
        print()


iit_reward_completion_length.plot_mean_length_iit_reward('settings_46', 700)
iit_reward_completion_length.plot_mean_length_iit_reward('settings_49', 1200)
iit_reward_completion_length.plot_mean_length_iit_reward('settings_64', 500)
iit_reward_completion_length.plot_mean_length_iit_reward('settings_65', 1200)
