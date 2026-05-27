from integrated_information_theory.enums_class import iit_log_type_enum, dataset_element_type_enum
from integrated_information_theory.logger.logger import logger


class training_logger(logger): 

    def __init__(self, log_file_name, log_type):
        super().__init__(log_file_name, has_token_details=False)

        self.log_type = log_type
        if self.log_type is None:
            raise Exception('log type is required')
        if iit_log_type_enum.TEST != self.log_type and iit_log_type_enum.TRAIN_TEST != self.log_type:
            raise Exception('log type has not been correctly determined')
        
    def add_to_buffer(self, log): 
        if self.buffer is None or log is None:
            return
        if iit_log_type_enum.TEST == self.log_type and dataset_element_type_enum.EVAL != log.get_split():
            return
        
        self.validate_log(log)

        filtered_list = list(filter(lambda x: x.equal(log) , self.buffer))
        if filtered_list is None or len(filtered_list) == 0:
            self.buffer.append(log)
            return 

        if len(filtered_list) > 1:
            raise Exception(f'Duplicate log{log}')

        found_item = filtered_list[0]
        if log.get_accuracy() is not None:
            found_item.set_accuracy(log.get_accuracy())
        if log.get_accuracy_reward() is not None:
            found_item.set_accuracy_reward(log.get_accuracy_reward())
        if log.get_phi_reward() is not None:
            found_item.set_phi_reward(log.get_phi_reward())
        if log.get_phi_reward_raw() is not None:
            found_item.set_phi_reward_raw(log.get_phi_reward_raw())
        if log.get_completion_loss() is not None:
            found_item.set_completion_loss(log.get_completion_loss())
        if log.get_perplexity() is not None:
            found_item.set_perplexity(log.get_perplexity())
        if log.get_entropy_reward() is not None:
            found_item.set_entropy_reward(log.get_entropy_reward())
        if log.get_token_count() is not None:
            found_item.set_token_count(log.get_token_count())
        if log.get_token_count_for_reduced_dim() is not None:
            found_item.set_token_count_for_reduced_dim(log.get_token_count_for_reduced_dim())
        if log.get_reduced_dim() is not None:
            found_item.set_reduced_dim(log.get_reduced_dim())
        if log.get_completion_embedding_shape() is not None:
            found_item.set_completion_embedding_shape(log.get_completion_embedding_shape())

    def convert_buffer(self): 
        list = []
        for log in self.buffer:
            b = {'Trainer_Global_Step' : log.get_trainer_global_step(), 
                'Split': log.get_split(), 
                'Sample_ID': log.get_sample_ID(), 
                'problem_id': log.get_problem_id(), 
                'Prompt': log.get_prompt(), 
                'Completion': log.get_completion(), 
                'Target': log.get_target(), 
                'Final_Answer': log.get_final_answer(), 
                'Accuracy_Reward': log.get_accuracy_reward(), 
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
                'Entropy_Reward': log.get_entropy_reward(),
                'Tpm_Loss': log.get_tpm_loss(),
                'Tpm_Entropy': log.get_tpm_entropy(),
                }
            list.append(b)            
        return list

    def get_fieldnames(self): 
        return ['Trainer_Global_Step', 
                'Split', 
                'Sample_ID', 
                'problem_id', 
                'Prompt', 
                'Completion', 
                'Target', 
                'Final_Answer', 
                'Accuracy_Reward', 
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
                'Entropy_Reward',
                'Tpm_Loss',
                'Tpm_Entropy',
                ]


# l = logger('./logs/test.csv', 'test')
# log = log_entity(sample_ID=1, split='test', trainer_global_step=1, prompt='Hello!', completion='how are you!')
# log.set_accuracy_reward(1)
# l.add_to_buffer(log)

# log = log_entity(sample_ID=1, split='test', trainer_global_step=1, prompt='Hello!', completion='how are you!')
# log.set_phi_reward(0.9)
# log.set_phi_reward_raw(2.0)
# l.add_to_buffer(log)

# log = log_entity(sample_ID=2, split='test', trainer_global_step=2, prompt='good morning!', completion='good morning!')
# log.set_accuracy_reward(1)
# l.add_to_buffer(log)

# log = log_entity(sample_ID=2, split='test', trainer_global_step=2, prompt='good morning!', completion='good morning!')
# log.set_phi_reward(0.8)
# log.set_phi_reward_raw(3.0)
# l.add_to_buffer(log)
# l.write_to_log_file()
