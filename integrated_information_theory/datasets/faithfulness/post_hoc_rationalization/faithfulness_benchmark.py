import os
import yaml
import pandas as pd

class faithfulness_benchmark(object):

    @staticmethod
    def process_yaml(data, topic):
        list = []
        for item in data['question_by_qid']:
            q = data['question_by_qid'][item]
            params = data['params']

            x = {}
            x['question_by_qid'] = item
            x['q_str'] = q['q_str']
            x['q_str_open_ended'] = q['q_str_open_ended']
            x['x_name'] = q['x_name']
            x['x_value'] = q['x_value']
            x['y_name'] = q['y_name']
            x['y_value'] = q['y_value']
            
            x['answer'] = params['answer']
            x['comparison'] = params['comparison']
            x['max_comparisons'] = params['max_comparisons']
            x['prop_id'] = params['prop_id']
            x['suffix'] = params['suffix'] if 'suffix' in params else ''
            x['uuid'] = params['uuid']
            x['topic'] = topic

            list.append(x)
        return list 

    @staticmethod
    def process_folder(folder_path):

        list = []
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith((".yaml", ".yml")):
                    topic = faithfulness_benchmark.extract_topic(filename)
                    if topic is None or len(topic) == 0: continue

                    full_path = os.path.join(root, filename)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f)

                        if data is None:
                            print(f"Skipping empty file: {full_path}")
                            continue

                        list.extend(faithfulness_benchmark.process_yaml(data, topic))

                    except yaml.YAMLError as e:
                        print(f"YAML error in {full_path}: {e}")
                    except Exception as e:
                        print(f"Error reading {full_path}: {e}")

        columns = ['question_by_qid', 'q_str', 'q_str_open_ended', 'x_name', 'x_value', 'y_name', 'y_value', 'answer', 'comparison', 'max_comparisons', 'prop_id', 'suffix', 'uuid', 'topic']
        df = pd.DataFrame(list, columns=columns)
        csv_file = f"{folder}.csv"
        df.to_csv(csv_file, index=False)

    def extract_topic(filename):
        if '_gt_' in filename: 
            return filename.split('_gt_')[0].replace('-', ' ')
        if '_lt_' in filename: 
            return filename.split('_lt_')[0].replace('-', ' ')
        
        return None

    def load_data(folder_path):
        df = pd.read_csv(f'{folder_path}.csv')
        pd.set_option('display.max_rows', None)
        df_result = df.groupby(['topic']).size().reset_index()
        print(df_result)



folder = "/home/hr_akbari/research/LLM_PostTraining/integrated_information_theory/datasets/data/faithfulness"
#faithfulness_benchmark.process_folder(folder)
faithfulness_benchmark.load_data(folder)
