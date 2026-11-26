from integrated_information_theory.logger.inference.inference_logger import inference_logger
from integrated_information_theory.logger.inference.iit_learnability.iit_learnability_log_entity import iit_learnability_log_entity
import pandas as pd 
import numpy as np 

import json

class iit_learnability_logger(inference_logger): 

    def __init__(self, log_file_name) -> None:
        super().__init__(log_file_name)

    def load_logs_list(self) -> list[iit_learnability_log_entity]:
        df = pd.read_csv(self.get_log_file_name())

        log_list: list[iit_learnability_log_entity] = []
        for _, a_row in df.iterrows():
            log = iit_learnability_log_entity()
            log.ID = a_row["ID"]
            log.split = a_row["Split"]
            log.sample_ID = a_row["Sample_ID"]
            log.problem_id = a_row["problem_id"]
            log.question = a_row["Question"]
            log.prompt = a_row["Prompt"]
            log.target = a_row["Target"]
            log.completion = a_row["Completion"]
            log.token_count = a_row["Token_Count"]
            log.final_answer = a_row["Final_Answer"]
            log.compared_final_answer = a_row["Compared_Final_Answer"]
            log.accuracy = a_row["Accuracy"]
            
            iit_completion_embedding_3 = a_row["IIT_Completion_Embedding_3"]
            if iit_completion_embedding_3 is not None:
                log.iit_completion_embedding_3 = np.array(json.loads(iit_completion_embedding_3),dtype=np.float64)

            iit_completion_embedding_4 = a_row["IIT_Completion_Embedding_4"]
            if iit_completion_embedding_4 is not None:
                log.iit_completion_embedding_4 = np.array(json.loads(iit_completion_embedding_4),dtype=np.float64)

            iit_completion_embedding_5 = a_row["IIT_Completion_Embedding_5"]
            if iit_completion_embedding_5 is not None:
                log.iit_completion_embedding_5 = np.array(json.loads(iit_completion_embedding_5),dtype=np.float64)

            completion_mean_representation = a_row["Completion_Mean_Representation"]
            if completion_mean_representation is not None:
                log.completion_mean_representation = np.array(json.loads(completion_mean_representation),dtype=np.float64)
            
            log_list.append(log)

        return log_list
    
    def convert_buffer(self): 
        list = []
        for log in self.buffer:
            b = { 
                'ID': log.ID, 
                'Split': log.split, 
                'Sample_ID': log.sample_ID, 
                'problem_id': log.problem_id, 
                'Question': log.question, 
                'Prompt': log.prompt, 
                'Target': log.target, 
                'Completion': log.completion, 
                'Token_Count': log.token_count, 
                'Final_Answer': log.final_answer, 
                'Compared_Final_Answer': log.compared_final_answer, 
                'Accuracy': log.accuracy, 
                
                'IIT_Completion_Embedding_3': json.dumps(log.iit_completion_embedding_3.tolist()) if log.iit_completion_embedding_3 is not None else None, 
                'IIT_Completion_Embedding_4': json.dumps(log.iit_completion_embedding_4.tolist()) if log.iit_completion_embedding_4 is not None else None, 
                'IIT_Completion_Embedding_5': json.dumps(log.iit_completion_embedding_5.tolist()) if log.iit_completion_embedding_5 is not None else None, 

                'Completion_Mean_Representation': json.dumps(log.completion_mean_representation.tolist()) if log.completion_mean_representation is not None else None, 
                }
            list.append(b)            
        return list

    def get_fieldnames(self): 
        return [ 
                'ID', 
                'Split', 
                'Sample_ID', 
                'problem_id', 
                'Question', 
                'Prompt', 
                'Target', 
                'Completion', 
                'Token_Count', 
                'Final_Answer', 
                'Compared_Final_Answer', 
                'Accuracy', 
                
                'IIT_Completion_Embedding_3', 
                'IIT_Completion_Embedding_4', 
                'IIT_Completion_Embedding_5', 

                'Completion_Mean_Representation', 
                ]

