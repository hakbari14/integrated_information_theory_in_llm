from integrated_information_theory.logger.inference.inference_log_entity import inference_log_entity

class inference_causalbench_log_entity(inference_log_entity):

    def __init__(self, ID, sample_ID, problem_id, split, prompt, target, question_type, scenario_ID):
        super().__init__(ID, sample_ID, problem_id, split, prompt, target)
        self.question_type = question_type
        self.scenario_ID = scenario_ID

    def get_question_type(self):
        return self.question_type

    def set_question_type(self, value):
        self.question_type = value

    def get_scenario_ID(self):
        return self.scenario_ID

    def set_scenario_ID(self, value):
        self.scenario_ID = value



