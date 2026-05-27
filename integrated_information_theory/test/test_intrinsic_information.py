import json
import numpy as np
import random
from tqdm import tqdm
from integrated_information_theory.intrinsic_information import intrinsic_information
from integrated_information_theory.entity.iit_entity import iit_entity
from integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from transformers import AutoTokenizer

np.seterr(divide='ignore', invalid='ignore')


class intrinsic_information_test: 

    def __init__(self, reduced_dim, calculation_type, tpm_creation_type, last_layer_computation_type, last_layer_computation_param, granularity, chunk_size=None):
        self.seed = 42

        config = intrinsic_information_config()
        config.set_calculation_type(calculation_type)
        config.set_adaptive_dim(False)
        config.set_reduced_dim(reduced_dim)
        config.set_tpm_creation_type(tpm_creation_type)
        config.set_last_layer_computation_type(last_layer_computation_type)
        config.set_last_layer_computation_param(last_layer_computation_param)
        config.set_granularity(granularity)
        config.set_chunk_size(chunk_size)

        self.src_path = './integrated_information_theory/test/'
        self.drive_path = f"{self.src_path}/intermediate_data/【Dataset_ToM_in_LLMs_and_Humans_large_scales】"
        self.outputs_encode_to_representations_path = f"{self.src_path}/intermediate_data/【outputs_encode_to_representations】"

        self.model_name = "Qwen2.5-3B-Instruct"
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tom_task = "ToM_B_Hinting"

        self.ToM_B_Task = json.load(open(f"{self.drive_path}/{self.tom_task}.json", "r"))
        self.ToM_B_Task_prompts = np.load(f"{self.outputs_encode_to_representations_path}/{self.tom_task}_prompts_{self.model_name}.npz")
        self.ToM_B_Task_responses = np.load(f"{self.outputs_encode_to_representations_path}/{self.tom_task}_responses_{self.model_name}.npz")
        self.intrinsic_information = intrinsic_information(config)

    def calculate(self):
        tqdm_prompt = tqdm(enumerate(self.ToM_B_Task["prompts"]),)

        iit_entity_list = []
        for prompt_index, prompt in tqdm_prompt:
            prompt_representation = self.ToM_B_Task_prompts[f"id_{prompt['id']}$sheet_{prompt['sheet']}"]

            filtered_list = list(filter(lambda x: x["id"] == prompt["id"] and x["sheet"] == prompt["sheet"], self.ToM_B_Task["responses"],))
            filtered_list.sort(key=lambda x: int(x["session"]))
            if len(filtered_list) == 0:
                raise Exception('filtered_list is empty')

            responses_raw = []
            for x in filtered_list:
                try:
                    to_append = {
                        "identifier": f"participant_{x['participant']}$session_{x['session']}$id_{x['id']}$sheet_{x['sheet']}$score_{x['score']}",
                        "response": f"{x['response']}",
                        "representation": self.ToM_B_Task_responses[f"participant_{x['participant']}$session_{x['session']}$id_{x['id']}$sheet_{x['sheet']}$score_{x['score']}"],
                    }
                except:
                    continue
                responses_raw.append(to_append)

            random.seed(self.seed + 42 * 1)
            np.random.seed(self.seed + 42 * 1)

            random.shuffle(responses_raw)
            random.shuffle(responses_raw)

            for idx, x in enumerate(responses_raw):
                prompt_emb = prompt_representation
                response_emb = x['representation']
                entity = iit_entity(idx)
                entity.set_promptID(prompt['text'])
                entity.set_prompt(prompt['text'])
                entity.set_completion(x['response'])
                entity.set_prompt_embedding(prompt_emb) 
                entity.set_completion_embedding(response_emb) 
                # entity.add_token_list(self.tokenizer, x['response'], response_emb)
                iit_entity_list.append(entity)
                
        iit_entity_list = self.intrinsic_information.calculate(iit_entity_list)
        iit_value_list = list(map(lambda x: x.get_iit_reward(), iit_entity_list))
        print()
        print(f'count = {len(iit_value_list)}')
        print(f'intrinsic information mean = {np.mean(iit_value_list)}')
        print(f'intrinsic information var = {np.var(iit_value_list)}')
        print(f'intrinsic information min = {np.min(iit_value_list)}')
        print(f'intrinsic information max = {np.max(iit_value_list)}')
        print()


c_sum = intrinsic_information_test(calculation_type='sum', reduced_dim=10, tpm_creation_type='batch', last_layer_computation_type='exp', last_layer_computation_param=0.09, granularity='token')
c_sum.calculate()

# print() 

# c_sum = intrinsic_information_test(calculation_type='sum', reduced_dim=4, tpm_creation_type='trajectory', last_layer_computation_type='exp', last_layer_computation_param=0.09, granularity='chunk', chunk_size=3)
# c_sum.calculate()

# print() 

# c_sum = intrinsic_information_test(calculation_type='sum', reduced_dim=4, tpm_creation_type='trajectory', last_layer_computation_type='exp', last_layer_computation_param=0.09,granularity='token')
# c_sum.calculate()
