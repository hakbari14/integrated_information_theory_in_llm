import pandas as pd
from tqdm import tqdm

class CompareBaselineAndIIT: 

    def __init__(self):
        self.similar_text = "You are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-by-step and justify your answer."
        self.converge_step_baseline = None
        self.converge_step_iit = None
        self.eval_baseline_path_file = None
        self.eval_iit_csv_path_file = None
        self.compare_csv_path_file = None


    def compare(self, df_baseline, df_iit):
        filtered_df_baseline = df_baseline[df_baseline['Trainer_Global_Step'] == self.converge_step_baseline]
        filtered_df_iit = df_iit[df_iit['Trainer_Global_Step'] == self.converge_step_iit]

        df = pd.DataFrame()
        df['sample_id'] = ''
        df['step_baseline'] = ''
        df['step_iit'] = ''
        df['prompt'] = ''
        df['completion_baseline'] = ''
        df['completion_iit'] = ''
        df['accuracy_reward'] = ''
        df['accuracy_reward_iit'] = ''
        df['reward'] = ''

        idx = 1
        for index_iit, row in tqdm(filtered_df_iit.iterrows()):
            sample_id = row['Sample_ID']
            step_iit = row['Trainer_Global_Step']
            prompt_iit = row['Prompt']
            response_iit = row['Completion']
            accuracy_reward_iit = row['Accuracy_Reward']
            reward = row['Phi_Reward_Raw']

            df_f = filtered_df_baseline[filtered_df_baseline.Sample_ID == sample_id]
            for index, row in df_f.iterrows():
                step_baseline = row['Trainer_Global_Step']
                prompt = row['Prompt']
                response = row['Completion']
                accuracy_reward = row['Accuracy_Reward']

                if prompt_iit != prompt:
                    continue

                df.loc[idx, 'sample_id'] = sample_id
                df.loc[idx, 'step_baseline'] = step_baseline
                df.loc[idx, 'step_iit'] = step_iit
                df.loc[idx, 'prompt'] = prompt
                df.loc[idx, 'completion_baseline'] = response
                df.loc[idx, 'completion_iit'] = response_iit
                df.loc[idx, 'accuracy_reward'] = accuracy_reward
                df.loc[idx, 'accuracy_reward_iit'] = accuracy_reward_iit
                df.loc[idx, 'reward'] = reward
                idx += 1

        df['step_baseline'] = df['step_baseline'].astype(int)
        df['step_iit'] = df['step_iit'].astype(int)
        df['accuracy_reward'] = df['accuracy_reward'].astype(int)
        df['accuracy_reward_iit'] = df['accuracy_reward_iit'].astype(int)
        df['reward'] = df['reward'].astype(float)
        return df

    def compare_and_write(self): 
        df_baseline = pd.read_csv(self.eval_baseline_path_file)
        df_iit = pd.read_csv(self.eval_iit_csv_path_file)
        df = self.compare(df_baseline, df_iit)
        df.to_csv(self.compare_csv_path_file, index=False)

    def calculate_statistics(self, df, accuracy_reward_value , accuracy_reward_value_iit):
        count = 0
        avg_baseline = 0.0
        avg_iit = 0.0
        avg_reward = 0.0
        similar_test_baseline_count = 0
        similar_test_iit_count = 0
        for index, row in df.iterrows():   
            if row['accuracy_reward'] != accuracy_reward_value or row['accuracy_reward_iit'] != accuracy_reward_value_iit:
                continue

            completion_baseline = row['completion_baseline']
            completion_iit = row['completion_iit']
            reward = row['reward']

            if isinstance(completion_baseline, str):
                if completion_baseline.find(self.similar_text) != -1:
                    similar_test_baseline_count += 1
                completion_baseline = completion_baseline.replace(self.similar_text, "")
            if isinstance(completion_iit, str):
                if completion_iit.find(self.similar_text) != -1:
                    similar_test_iit_count += 1
                completion_iit = completion_iit.replace(self.similar_text, "")

            count = count + 1
            avg_reward = avg_reward + reward
            if isinstance(completion_baseline, str):
                avg_baseline = avg_baseline + len(completion_baseline)
            if isinstance(completion_iit, str):
                avg_iit = avg_iit + len(completion_iit)

        avg_baseline /= count
        avg_iit /= count
        avg_reward /= count

        return count, avg_baseline, avg_iit, avg_reward, similar_test_baseline_count, similar_test_iit_count

    def show_statistics(self):
        df = pd.read_csv(self.compare_csv_path_file)
        
        count_TT, avg_baseline_TT, avg_iit_TT, avg_reward, similar_test_baseline_count, similar_test_iit_count = self.calculate_statistics(df, accuracy_reward_value = 1,accuracy_reward_value_iit = 1)
        recall_baseline_length = avg_baseline_TT * count_TT
        recall_iit_length = avg_iit_TT * count_TT

        print("Both(Baseline and IIT) are correct")
        print("Count: ", count_TT)
        print("similar_test_baseline_count: ", similar_test_baseline_count)
        print("similar_test_iit_count: ", similar_test_iit_count)
        print("Avg Baseline: ", avg_baseline_TT)
        print("Avg IIT: ", avg_iit_TT)
        print("Avg Reward: ", avg_reward)
        print("Percent : ", 100* round(avg_iit_TT / avg_baseline_TT, 2))

        count_TT, avg_baseline_TT, avg_iit_TT, avg_reward, similar_test_baseline_count, similar_test_iit_count = self.calculate_statistics(df, accuracy_reward_value = 0,accuracy_reward_value_iit = 1)
        recall_baseline_length += avg_baseline_TT * count_TT
        recall_iit_length += avg_iit_TT * count_TT
        
        print("Baseline is incorrect and IIT is correct")
        print("Count: ", count_TT)
        print("similar_test_baseline_count: ", similar_test_baseline_count)
        print("similar_test_iit_count: ", similar_test_iit_count)
        print("Avg Baseline: ", avg_baseline_TT)
        print("Avg IIT: ", avg_iit_TT)
        print("Avg Reward: ", avg_reward)
        print("Percent : ", 100* round(avg_iit_TT / avg_baseline_TT, 2))

        count_TT, avg_baseline_TT, avg_iit_TT, avg_reward, similar_test_baseline_count, similar_test_iit_count = self.calculate_statistics(df, accuracy_reward_value = 1,accuracy_reward_value_iit = 0)
        print("Baseline is correct and IIT is incorrect")
        print("Count: ", count_TT)
        print("similar_test_baseline_count: ", similar_test_baseline_count)
        print("similar_test_iit_count: ", similar_test_iit_count)
        print("Avg Baseline: ", avg_baseline_TT)
        print("Avg IIT: ", avg_iit_TT)
        print("Avg Reward: ", avg_reward)
        print("Percent : ", 100* round(avg_iit_TT / avg_baseline_TT, 2))

        count_TT, avg_baseline_TT, avg_iit_TT, avg_reward, similar_test_baseline_count, similar_test_iit_count = self.calculate_statistics(df, accuracy_reward_value = 0,accuracy_reward_value_iit = 0)
        print("Both(Baseline and IIT) are incorrect")
        print("Count: ", count_TT)
        print("similar_test_baseline_count: ", similar_test_baseline_count)
        print("similar_test_iit_count: ", similar_test_iit_count)
        print("Avg Baseline: ", avg_baseline_TT)
        print("Avg IIT: ", avg_iit_TT)
        print("Avg Reward: ", avg_reward)
        print("Percent : ", 100* round(avg_iit_TT / avg_baseline_TT, 2))

        print("Recall Correct Answer : ", 100* round(recall_iit_length / recall_baseline_length, 2))


    def show_dataframe(self, df):
        pd.set_option("display.max_rows", None)    
        pd.set_option('display.width', None)
        filtered_df_compare_both_correct = df[(df['accuracy_reward'] == 1) & (df['accuracy_reward_iit'] == 1)]
        print(filtered_df_compare_both_correct)

