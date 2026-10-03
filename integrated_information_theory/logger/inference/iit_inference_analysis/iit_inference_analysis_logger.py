from integrated_information_theory.logger.inference.inference_logger import inference_logger

class iit_inference_analysis_logger(inference_logger): 

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
                
                'IIR_Reward_Raw_Actual': log.iir_reward_raw_actual, 
                'IIR_Reward_Raw': log.iir_reward_raw, 
                'IIR_Reward': log.iir_reward, 
                'Completion_Embedding_Shape_IIR': log.completion_embedding_shape_iir, 

                'Phi_S_Reward_Raw_Actual': log.phi_s_reward_raw_actual, 
                'Phi_S_Reward_Raw': log.phi_s_reward_raw, 
                'Phi_S_Reward': log.phi_s_reward, 
                'Completion_Embedding_Shape_Phi_S': log.completion_embedding_shape_phi_s, 
                
                'Phi_Reward_Raw_Actual': log.phi_reward_raw_actual, 
                'Phi_Reward_Raw': log.phi_reward_raw, 
                'Phi_Reward': log.phi_reward, 
                'Completion_Embedding_Shape_Phi': log.completion_embedding_shape_phi, 
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
                
                'IIR_Reward_Raw_Actual', 
                'IIR_Reward_Raw', 
                'IIR_Reward', 
                'Completion_Embedding_Shape_IIR', 
                
                'Phi_S_Reward_Raw_Actual', 
                'Phi_S_Reward_Raw', 
                'Phi_S_Reward', 
                'Completion_Embedding_Shape_Phi_S', 
                
                'Phi_Reward_Raw_Actual', 
                'Phi_Reward_Raw', 
                'Phi_Reward', 
                'Completion_Embedding_Shape_Phi', 
                ]

