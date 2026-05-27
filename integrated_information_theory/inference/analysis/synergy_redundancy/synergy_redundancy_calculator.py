from abc import ABC, abstractmethod
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from integrated_information_theory.llm_representation import llm_representation
from tqdm import tqdm
import torch
import gc
import pandas as pd
from peft import PeftModel
import re
import numpy as np


class synergy_redundancy_calculator(ABC): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        self.model_name = model_name
        if self.model_name is None:
            raise Exception('model name is required')
        self.peft_checkpoint_path = peft_checkpoint_path

        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = getattr(torch, "bfloat16"),
            bnb_4bit_use_double_quant = False,
        )
        if self.peft_checkpoint_path != None :
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config = bnb_config)
            self.model = PeftModel.from_pretrained(base_model, self.peft_checkpoint_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config = bnb_config)

        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.representation = llm_representation()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.inference_mode()
    def calculate(self): 
        try:
            df = pd.read_csv(self.get_log_file_name())

            if 'Synergy' not in df.columns:
                df['Synergy'] = np.nan
                df['Synergy'] = df['Synergy'].astype('float64')

            if 'Redundancy' not in df.columns:
                df['Redundancy'] = np.nan             
                df['Redundancy'] = df['Redundancy'].astype('float64')

            for index, row in tqdm(enumerate(df.iterrows()), bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
                completion = df.loc[index, "Completion"]
                completion_emb, _ = self.representation.extract_representation_last_layer(completion, self.model, self.tokenizer)

                sentences = self.get_all_sentences(completion)
                sentence_emb_list = []
                for sentence in sentences: 
                    sentence_emb, _ = self.representation.extract_representation_last_layer(sentence, self.model, self.tokenizer)
                    sentence_emb_list.append(sentence_emb)

                df.at[index, "Synergy"] = self.calculate_synergy(completion_emb, sentence_emb_list)
                df.at[index, "Redundancy"] = self.calculate_redundancy(sentence_emb_list)

                gc.collect()
                torch.cuda.empty_cache()

            df.to_csv(self.get_log_file_name(), index=False)            
        except Exception as e:
            print(f"Exception : {e}")

        return None

    def calculate_synergy(self, completion_emb, sentence_emb_list):
        similarity_completion = self.calculate_embedding_similarity(completion_emb, completion_emb)

        sum_similarity_sentences = 0
        for sentence_emb in sentence_emb_list: 
            sum_similarity_sentences += self.calculate_embedding_similarity(completion_emb, sentence_emb)

        return similarity_completion - sum_similarity_sentences / len(sentence_emb_list)

    def calculate_redundancy(self, sentence_emb_list):
        sum_similarity_sentences = 0
        similarity_no = 0
        for i in range(len(sentence_emb_list)): 
            embedding_i = sentence_emb_list[i]
            for j in range(i + 1, len(sentence_emb_list)): 
                embedding_j = sentence_emb_list[j]
                sum_similarity_sentences += self.calculate_embedding_similarity(embedding_i, embedding_j)
                similarity_no += 1

        return sum_similarity_sentences / similarity_no

    def calculate_embedding_similarity(self, embedding_1, embedding_2):
        norm_1 = embedding_1 / np.linalg.norm(embedding_1)
        norm_2 = embedding_2 / np.linalg.norm(embedding_2)
        
        similarity = np.dot(norm_1, norm_2)
        return similarity

    def get_all_sentences(self, completion):
        if completion is None: 
            return []
        return re.split(r'(?<=[.!?])\s+', completion.strip())

    @abstractmethod
    def get_log_file_name(self):
        pass

