from integrated_information_theory.logger.inference.inference_log_entity import inference_log_entity

class inference_reward_hacking_log_entity(inference_log_entity):

    def __init__(self, ID, sample_ID, problem_id, split, prompt, target):
        super().__init__(ID, sample_ID, problem_id, split, prompt, target)
        self.completion_portion_size = None

    def validate(self): 
        super().validate()
        if self.completion_portion_size is None :
            raise Exception('completion_portion_size is required')

    def get_completion_portion_size(self):
        return self.completion_portion_size

    def set_completion_portion_size(self, value):
        self.completion_portion_size = value


