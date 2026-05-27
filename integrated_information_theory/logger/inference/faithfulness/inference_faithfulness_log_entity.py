from integrated_information_theory.logger.inference.inference_log_entity import inference_log_entity

class inference_faithfulness_log_entity(inference_log_entity):

    def __init__(self, ID, sample_ID, problem_id, split, prompt, target, prompt_2, question_by_qid):
        super().__init__(ID, sample_ID, problem_id, split, prompt, target)
        self.prompt_2 = prompt_2
        self.question_by_qid = question_by_qid
        self.completion_2 = None
        self.final_answer_2 = None

    def get_prompt_2(self):
        return self.prompt_2

    def set_prompt_2(self, value):
        self.prompt_2 = value

    def get_completion_2(self):
        return self.completion_2

    def set_completion_2(self, value):
        self.completion_2 = value

    def get_final_answer_2(self):
        return self.final_answer_2

    def set_final_answer_2(self, value):
        self.final_answer_2 = value

    def get_question_by_qid(self):
        return self.question_by_qid

    def set_question_by_qid(self, value):
        self.question_by_qid = value


