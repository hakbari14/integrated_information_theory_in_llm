from integrated_information_theory.logger.inference.inference_logger import inference_logger
import json

class iit_learnability_logger(inference_logger): 

    def __init__(self, log_file_name) -> None:
        super().__init__(log_file_name)

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
                
                'IIT_Reward_Settings': log.iit_reward_settings, 
                'IIT_Reward_Raw_Actual': log.iit_reward_raw_actual, 
                'IIT_Reward_Raw': log.iit_reward_raw, 
                'IIT_Reward': log.iit_reward, 
                'Completion_Embedding_Shape_IIT': log.completion_embedding_shape_iit, 

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
                
                'IIT_Reward_Settings', 
                'IIT_Reward_Raw_Actual', 
                'IIT_Reward_Raw', 
                'IIT_Reward', 
                'Completion_Embedding_Shape_IIT', 

                'Completion_Mean_Representation', 
                ]

