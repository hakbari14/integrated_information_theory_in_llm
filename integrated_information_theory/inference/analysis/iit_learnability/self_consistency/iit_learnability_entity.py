from dataclasses import dataclass, field
from typing import Optional, List
from integrated_information_theory.inference.analysis.iit_learnability.self_consistency.iit_learnability_detail_entity import iit_learnability_detail_entity

@dataclass
class iit_learnability_entity:

    ID : Optional[str] = None
    sample_ID : Optional[str] = None
    problem_id : Optional[str] = None
    split : Optional[str] = None
    question : Optional[str] = None
    prompt : Optional[str] = None
    target : Optional[str] = None

    consistency_list: List[iit_learnability_detail_entity] = field(default_factory=list)

    def add_consistency_list(self, log_detail: iit_learnability_detail_entity):
        self.consistency_list.append(log_detail)

