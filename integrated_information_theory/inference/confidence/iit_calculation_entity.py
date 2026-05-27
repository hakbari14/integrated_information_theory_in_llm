from pydantic import BaseModel
from typing import List

class self_consistency_log_detail_api_entity(BaseModel):
    index: str
    completion: str
    final_answer: str
    compared_final_answer: str
    token_count: int
    accuracy: bool

    def get_index(self) -> str:
        return self.index
    def set_index(self, index: str) -> None:
        self.index = index

    def get_completion(self) -> str:
        return self.completion
    def set_completion(self, completion: str) -> None:
        self.completion = completion

    def get_final_answer(self) -> str:
        return self.final_answer
    def set_final_answer(self, final_answer: str) -> None:
        self.final_answer = final_answer

    def get_compared_final_answer(self) -> str:
        return self.compared_final_answer
    def set_compared_final_answer(self, compared_final_answer: str) -> None:
        self.compared_final_answer = compared_final_answer

    def get_token_count(self) -> int:
        return self.token_count
    def set_token_count(self, token_count: int) -> None:
        self.token_count = token_count

    def get_accuracy(self) -> bool:
        return self.accuracy
    def set_accuracy(self, accuracy: bool) -> None:
        self.accuracy = accuracy
        

class self_consistency_log_api_entity(BaseModel):
    sample_ID: str
    problem_id: str
    split: str
    prompt: str
    target: str
    completion: str
    token_count: int
    final_answer: str
    accuracy: bool
    completion_embedding_shape: str
    completion_loss: float
    perplexity: float
    entropy: float
    token_count_for_reduced_dim: str
    reduced_dim: int
    phi_reward: float
    phi_reward_raw: float
    phi_reward_raw_actual: float
    tpm_loss: float
    tpm_entropy: float
    consistency_list : List[self_consistency_log_detail_api_entity]
    
    def get_sample_ID(self) -> str:
        return self.sample_ID
    def set_sample_ID(self, sample_ID: str) -> None:
        self.sample_ID = sample_ID
    
    def get_problem_id(self) -> str:
        return self.problem_id
    def set_problem_id(self, problem_id: str) -> None:
        self.problem_id = problem_id

    def get_split(self) -> str:
        return self.split
    def set_split(self, split: str) -> None:
        self.split = split

    def get_prompt(self) -> str:
        return self.prompt
    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

    def get_target(self) -> str:
        return self.target
    def set_target(self, target: str) -> None:
        self.target = target

    def get_completion(self) -> str:
        return self.completion
    def set_completion(self, completion: str) -> None:
        self.completion = completion

    def get_token_count(self) -> int:
        return self.token_count
    def set_token_count(self, token_count: int) -> None:
        self.token_count = token_count

    def get_final_answer(self) -> str:
        return self.final_answer
    def set_final_answer(self, final_answer: str) -> None:
        self.final_answer = final_answer

    def get_accuracy(self) -> bool:
        return self.accuracy
    def set_accuracy(self, accuracy: bool) -> None:
        self.accuracy = accuracy

    def get_completion_embedding_shape(self) -> str:
        return self.completion_embedding_shape
    def set_completion_embedding_shape(self, completion_embedding_shape: str) -> None:
        self.completion_embedding_shape = completion_embedding_shape

    def get_perplexity(self) -> float:
        return self.perplexity
    def set_perplexity(self, perplexity: float) -> None:
        self.perplexity = perplexity

    def get_entropy(self) -> float:
        return self.entropy
    def set_entropy(self, entropy: float) -> None:
        self.entropy = entropy

    def get_token_count_for_reduced_dim(self) -> str:
        return self.token_count_for_reduced_dim
    def set_token_count_for_reduced_dim(self, token_count_for_reduced_dim: str) -> None:
        self.token_count_for_reduced_dim = token_count_for_reduced_dim

    def get_reduced_dim(self) -> int:
        return self.reduced_dim
    def set_reduced_dim(self, reduced_dim: int) -> None:
        self.reduced_dim = reduced_dim
        
    def get_phi_reward(self) -> float:
        return self.phi_reward
    def set_phi_reward(self, phi_reward: float) -> None:
        self.phi_reward = phi_reward

    def get_phi_reward_raw(self) -> float:
        return self.phi_reward_raw
    def set_phi_reward_raw(self, phi_reward_raw: float) -> None:
        self.phi_reward_raw = phi_reward_raw

    def get_phi_reward_raw_actual(self) -> float:
        return self.phi_reward_raw_actual
    def set_phi_reward_raw_actual(self, phi_reward_raw_actual: float) -> None:
        self.phi_reward_raw_actual = phi_reward_raw_actual

    def get_tpm_loss(self) -> float:
        return self.tpm_loss
    def set_tpm_loss(self, tpm_loss: float) -> None:
        self.tpm_loss = tpm_loss

    def get_tpm_entropy(self) -> float:
        return self.tpm_entropy
    def set_tpm_entropy(self, tpm_entropy: float) -> None:
        self.tpm_entropy = tpm_entropy

    def get_consistency_list(self) -> List[self_consistency_log_detail_api_entity]:
        return self.consistency_list
    def set_consistency_list(self, consistency_list: List[self_consistency_log_detail_api_entity]) -> None:
        self.consistency_list = consistency_list
    def add_consistency_list(self, item: self_consistency_log_detail_api_entity) -> None:
        self.consistency_list.append(item)

class self_consistency_log_res_api_entity(self_consistency_log_api_entity):
    ii_calculator_name: str
    
    def get_ii_calculator_name(self) -> str:
        return self.ii_calculator_name
    def set_ii_calculator_name(self, ii_calculator_name: str) -> None:
        self.ii_calculator_name = ii_calculator_name


