from integrated_information_theory.entity.iit_token_entity import iit_token_entity
from integrated_information_theory.utils import my_utils
import numpy as np
import pyphi
import gc

class iit_entity:

    def __init__(self, key):
        self.key = key
        if self.key is None:
            raise Exception('key is required')
        
        self.iit_reward = 0.0
        self.iit_reward_raw = 0.0
        self.iit_reward_raw_actual = 0.0
        self.tpm_loss = None
        self.tpm_entropy = None
        self.promptID = None
        self.prompt = None
        self.completion = None
        self.token_count = None
        self.prompt_embedding = None
        self.completion_loss = None
        self.completion_entropy = None
        self.completion_embedding = None
        self.completion_embedding_shape = None
        self.completion_embedding_for_pca = None
        self.completion_concatenated_embedding = None
        self.markov_chain = None
        self.token_count_for_reduced_dim = None
        self.reduced_dim = None
        self.iit_token_list = []
 
    def get_key(self):
        return self.key

    def set_key(self, value):
        self.key = value

    def get_promptID(self):
        return self.promptID

    def set_promptID(self, value):
        self.promptID = value

    def get_prompt(self):
        return self.prompt

    def set_prompt(self, value):
        self.prompt = value

    def get_completion(self):
        return self.completion

    def set_completion(self, value):
        self.completion = value

    def get_token_count(self):
        return self.token_count

    def set_token_count(self, value):
        self.token_count = value

    def get_prompt_embedding(self):
        return self.prompt_embedding

    def set_prompt_embedding(self, value):
        self.prompt_embedding = value

    def get_completion_loss(self):
        return self.completion_loss

    def set_completion_loss(self, value):
        self.completion_loss = value

    def get_completion_entropy(self):
        return self.completion_entropy

    def set_completion_entropy(self, value):
        self.completion_entropy = value

    def get_completion_embedding(self):
        return self.completion_embedding

    def set_completion_embedding(self, value):
        self.completion_embedding = value

    def get_completion_embedding_shape(self):
        return self.completion_embedding_shape

    def set_completion_embedding_shape(self, value):
        self.completion_embedding_shape = value

    def get_completion_embedding_for_pca(self):
        return self.completion_embedding_for_pca

    def set_completion_embedding_for_pca(self, value):
        self.completion_embedding_for_pca = value

    def get_iit_reward(self):
        return self.iit_reward

    def set_iit_reward(self, value):
        self.iit_reward = value

    def get_iit_reward_raw(self):
        return self.iit_reward_raw

    def set_iit_reward_raw(self, value):
        self.iit_reward_raw = value

    def get_iit_reward_raw_actual(self):
        return self.iit_reward_raw_actual

    def set_iit_reward_raw_actual(self, value):
        self.iit_reward_raw_actual = value

    def get_tpm_loss(self):
        return self.tpm_loss

    def set_tpm_loss(self, value):
        self.tpm_loss = value

    def get_tpm_entropy(self):
        return self.tpm_entropy

    def set_tpm_entropy(self, value):
        self.tpm_entropy = value

    def get_completion_concatenated_embedding(self):
        return self.completion_concatenated_embedding

    def set_completion_concatenated_embedding(self, value):
        self.completion_concatenated_embedding = value

    def get_markov_chain(self):
        return self.markov_chain

    def set_markov_chain(self, value):
        self.markov_chain = value

    def get_iit_token_list(self):
        return self.iit_token_list

    def set_iit_token_list(self, value):
        self.iit_token_list = value

    def add_iit_token_list(self, value):
        self.iit_token_list.append(value)

    def get_token_count_for_reduced_dim(self): 
        return self.token_count_for_reduced_dim

    def set_token_count_for_reduced_dim(self, value): 
        self.token_count_for_reduced_dim = value

    def get_reduced_dim(self): 
        return self.reduced_dim

    def set_reduced_dim(self, value): 
        self.reduced_dim = value

    def set_completion_embedding_and_shape(self, completion_embedding):
        self.set_completion_embedding(completion_embedding)
        if completion_embedding is not None: 
            self.set_completion_embedding_shape(completion_embedding.shape)

    def add_token_list(self, tokenizer, completion, completion_emb):
        tokens = tokenizer.tokenize(completion)
        if my_utils.has_add_bos_token(tokenizer):
            tokens.insert(0, 'BOS')

        if len(tokens) != completion_emb.shape[1]:
            raise Exception(f'The number of tokens is not the same as the representation dimensions, completion_shape ={completion_emb.shape}, token count = {len(tokens)}')

        self.set_token_count(completion_emb.shape[1])
        
        for idx, t in enumerate(tokens): 
            clean_token = t.replace('▁', ' ').replace('Ġ', ' ')
            clean_token = clean_token.replace(chr(269), '').replace(chr(266), '')
            # Skip empty tokens
            if not clean_token or clean_token.startswith('�'): continue
            t_entity = iit_token_entity(idx, clean_token)
            self.add_iit_token_list(t_entity)
        
        return None

    def aggregate_token_list(self, start, length):
        result = []
        end = start + length
        for t in self.get_iit_token_list(): 
            if t.get_token_number() >= start and t.get_token_number() < end:  
                result.append(t)

        if len(result) == 0:
            return '', ''
        
        token_text_list = list(map(lambda x: x.get_token(), result))
        token_emb_list = list(map(lambda x: x.get_attended_embedding(), result))
        return " ".join(token_text_list), np.concatenate(tuple(token_emb_list), axis=0,)

    def set_all_markov_chain(self, value):
        self.markov_chain = value
        for idx, m in enumerate(self.markov_chain):
            t_entity_list = list(filter(lambda x: x.get_token_number() == idx , self.get_iit_token_list()))
            if len(t_entity_list) == 0: continue
            if len(t_entity_list) != 1:
                raise Exception(f'token Entity not found by key {idx} and size list {len(t_entity_list)}')
            
            t_entity = t_entity_list[0]
            t_entity.set_token_markov_chain_0_1_emb(m)
            t_entity.set_state_index(pyphi.convert.state2le_index(m))

    def set_integrated_information_value(self, state, value):
        for t in self.get_iit_token_list(): 
            if t.get_state_index() != state:  
                continue
            
            t.set_iit_value(value)

    def set_intrinsic_information_value(self, state, ii_value, effect_value, cause_value, effect_state_index, cause_state_index):
        for t in self.get_iit_token_list(): 
            if t.get_state_index() != state:  
                continue

            t.set_iit_effect_value(effect_value)
            t.set_iit_cause_value(cause_value)
            t.set_iit_value(ii_value)
            t.set_iit_effect_state_index(effect_state_index)
            t.set_iit_cause_state_index(cause_state_index)
            
        return None

    def is_calcutable(self):
        if self.completion is None or len(self.completion) == 0:
            return False
        if self.prompt_embedding is None or self.prompt_embedding.shape is None:
            return False
        if self.completion_embedding is None or self.completion_embedding.shape is None:
            return False
        
        return True

    def has_completion_embedding_for_pca(self): 
        return self.completion_embedding_for_pca is not None and self.completion_embedding_for_pca.shape is not None

    def has_concatenated_embedding(self): 
        return self.completion_concatenated_embedding is not None and self.completion_concatenated_embedding.shape is not None

    @staticmethod
    def clone_list(entity_list): 
        new_entity_list = []
        for entity in entity_list: 
            new_entity_list.append(iit_entity.clone(entity))
            
        return new_entity_list

    @staticmethod
    def clone(entity): 
        new_entity = iit_entity(entity.get_key())
        new_entity.set_promptID(entity.get_promptID())
        new_entity.set_prompt(entity.get_prompt())
        new_entity.set_completion(entity.get_completion())
        new_entity.set_token_count(entity.get_token_count())
        if entity.get_prompt_embedding() is not None: 
            new_entity.set_prompt_embedding(entity.get_prompt_embedding().copy())
        new_entity.set_completion_loss(entity.get_completion_loss())
        new_entity.set_completion_entropy(entity.get_completion_entropy())
        if entity.get_completion_embedding() is not None: 
            new_entity.set_completion_embedding(entity.get_completion_embedding().copy())
        if entity.get_completion_embedding_for_pca() is not None: 
            new_entity.set_completion_embedding_for_pca(entity.get_completion_embedding_for_pca().copy())
        if entity.get_completion_concatenated_embedding() is not None: 
            new_entity.set_completion_concatenated_embedding(entity.get_completion_concatenated_embedding().copy())
        new_entity.set_markov_chain(entity.get_markov_chain())
        new_entity.set_token_count_for_reduced_dim(entity.get_token_count_for_reduced_dim())
        new_entity.set_reduced_dim(entity.get_reduced_dim())
        new_entity.set_iit_reward(entity.get_iit_reward())
        new_entity.set_iit_reward_raw(entity.get_iit_reward_raw())
        new_entity.set_iit_reward_raw_actual(entity.get_iit_reward_raw_actual())

        for token_entity in entity.get_iit_token_list():
            new_token_entity = iit_token_entity.clone(token_entity)
            new_entity.add_iit_token_list(new_token_entity)
    
        return new_entity

    @staticmethod
    def release_memory(entity_list): 
        for entity in entity_list: 
            entity.set_prompt_embedding(None)
            entity.set_completion_embedding(None)
            entity.set_completion_embedding_for_pca(None)
            entity.set_completion_concatenated_embedding(None)
            entity.set_markov_chain(None)
            
            for token_entity in entity.get_iit_token_list():
                token_entity.set_attended_embedding(None)
                token_entity.set_token_markov_chain_0_1_emb(None)
                
        gc.collect()




