from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class iit_inference_analysis_log_entity:

    ID : Optional[str] = None
    sample_ID : Optional[str] = None
    problem_id : Optional[str] = None
    split : Optional[str] = None
    question : Optional[str] = None
    prompt : Optional[str] = None
    target : Optional[str] = None

    index : Optional[str] = None
    completion : Optional[str] = None
    token_count : Optional[int] = None
    final_answer : Optional[str] = None
    compared_final_answer : Optional[str] = None
    accuracy : Optional[bool] = None

    iir_reward_raw_actual : Optional[float] = 0.0
    iir_reward_raw : Optional[float] = 0.0
    iir_reward : Optional[float] = 0.0
    completion_embedding_shape_iir : Optional[str] = None
    
    phi_s_reward_raw_actual : Optional[float] = 0.0
    phi_s_reward_raw : Optional[float] = 0.0
    phi_s_reward : Optional[float] = 0.0
    completion_embedding_shape_phi_s : Optional[str] = None
    
    phi_reward_raw_actual : Optional[float] = 0.0
    phi_reward_raw : Optional[float] = 0.0
    phi_reward : Optional[float] = 0.0
    completion_embedding_shape_phi : Optional[str] = None

    def equal(self, x):
        return self.ID == x.ID

    def validate(self): 
        if self.ID is None:
            raise Exception('ID is required')
        if self.sample_ID is None:
            raise Exception('Sample ID is required')
        if self.prompt is None or len(self.prompt) == 0:
            raise Exception('prompt is required')
        if self.split is None:
            raise Exception('split is required')
        if self.target is None:
            raise Exception('target is required')
        
