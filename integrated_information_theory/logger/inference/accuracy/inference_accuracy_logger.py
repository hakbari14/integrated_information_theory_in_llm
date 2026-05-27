from integrated_information_theory.logger.inference.inference_logger import inference_logger


class inference_accuracy_logger(inference_logger): 

    def __init__(self, log_file_name, has_token_details = False):
        super().__init__(log_file_name, has_token_details=has_token_details)

    def convert_buffer(self): 
        list = []
        for log in self.buffer:
            b = { 
                'ID': log.get_ID(), 
                'Split': log.get_split(), 
                'Sample_ID': log.get_sample_ID(), 
                'problem_id': log.get_problem_id(), 
                'Prompt': log.get_prompt(), 
                'Target': log.get_target(), 
                'Completion': log.get_completion(), 
                'Final_Answer': log.get_final_answer(),
                'Accuracy': log.get_accuracy(),
                'Token_Count': log.get_token_count(), 
                'Token_Count_Reduced_Dimention': log.get_token_count_for_reduced_dim(), 
                'Reduced_Dimention': log.get_reduced_dim(), 
                'Phi_Reward': log.get_phi_reward(),
                'Phi_Reward_Raw': log.get_phi_reward_raw(),
                'Phi_Reward_Raw_Actual': log.get_phi_reward_raw_actual(),
                'Completion_Embedding_Shape': log.get_completion_embedding_shape(), 
                'Completion_Loss': log.get_completion_loss(), 
                'Perplexity': log.get_perplexity(),
                'Entropy': log.get_entropy(),
                'Tpm_Loss': log.get_tpm_loss(),
                'Tpm_Entropy': log.get_tpm_entropy(),
                }
            list.append(b)            
        return list

    def get_fieldnames(self): 
        return [ 
                'ID',
                'Split', 
                'Sample_ID', 
                'problem_id', 
                'Prompt', 
                'Target', 
                'Completion', 
                'Final_Answer', 
                'Accuracy',
                'Token_Count',
                'Token_Count_Reduced_Dimention',
                'Reduced_Dimention',
                'Phi_Reward',
                'Phi_Reward_Raw',
                'Phi_Reward_Raw_Actual',
                'Completion_Embedding_Shape',
                'Completion_Loss',
                'Perplexity',
                'Entropy',
                'Tpm_Loss',
                'Tpm_Entropy',
                ]

