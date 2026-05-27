from integrated_information_theory.logger.inference.inference_log_entity import inference_log_entity

class inference_code_log_entity(inference_log_entity):

    def __init__(self, ID, sample_ID, problem_id, split, prompt, target):
        super().__init__(ID, sample_ID, problem_id, split, prompt, target)

