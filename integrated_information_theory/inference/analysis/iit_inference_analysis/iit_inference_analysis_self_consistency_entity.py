from dataclasses import dataclass
from typing import Optional

@dataclass
class iit_inference_analysis_self_consistency_entity:

    index : Optional[str] = None
    completion : Optional[str] = None
    token_count : Optional[int] = None
    final_answer : Optional[str] = None
    compared_final_answer : Optional[str] = None
    accuracy : Optional[bool] = None

    
