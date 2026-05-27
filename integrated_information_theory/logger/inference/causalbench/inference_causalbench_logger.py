from integrated_information_theory.logger.inference.inference_logger import inference_logger


class inference_causalbench_logger(inference_logger): 

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
                'Question_Type': log.get_question_type(), 
                'Scenario_ID': log.get_scenario_ID(),
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
                'Question_Type',
                'Scenario_ID',
                ]

