from integrated_information_theory.logger.inference.inference_logger import inference_logger


class inference_faithfulness_logger(inference_logger): 

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
                'Prompt_1': log.get_prompt(), 
                'Prompt_2': log.get_prompt_2(), 
                'Answer': log.get_target(), 
                'Completion_1': log.get_completion(), 
                'Final_Answer_1': log.get_final_answer(),
                'Completion_2': log.get_completion_2(), 
                'Final_Answer_2': log.get_final_answer_2(),
                'Accuracy': log.get_accuracy(),
                'Question_By_Qid': log.get_question_by_qid(), 
                }
            list.append(b)            
        return list

    def get_fieldnames(self): 
        return [ 
                'ID',
                'Split', 
                'Sample_ID', 
                'problem_id', 
                'Prompt_1', 
                'Prompt_2', 
                'Answer', 
                'Completion_1', 
                'Final_Answer_1', 
                'Completion_2', 
                'Final_Answer_2', 
                'Accuracy',
                'Question_By_Qid',
                ]

