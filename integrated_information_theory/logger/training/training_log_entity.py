from integrated_information_theory.logger.log_entity import log_entity

class training_log_entity(log_entity):

    def __init__(self, sample_ID, problem_id, split, trainer_global_step, prompt, target, completion):
        super().__init__(sample_ID, problem_id, split, prompt, target)
        self.trainer_global_step = trainer_global_step
        self.completion = completion
        self.accuracy_reward = None
        self.entropy_reward = None
        self.adaptive_length_reward = None

    def validate(self): 
        super().validate()

    def equal(self, x):
        if self.sample_ID != x.get_sample_ID():
            return False
        if self.split != x.get_split():
            return False
        if self.completion != x.get_completion():
            return False
        
        return True

    def get_trainer_global_step(self):
        return self.trainer_global_step

    def set_trainer_global_step(self, value):
        self.trainer_global_step = value

    def get_accuracy_reward(self):
        return self.accuracy_reward

    def set_accuracy_reward(self, value):
        self.accuracy_reward = value

    def get_entropy_reward(self):
        return self.entropy_reward

    def set_entropy_reward(self, value):
        self.entropy_reward = value

    def get_adaptive_length_reward(self):
        return self.adaptive_length_reward

    def set_adaptive_length_reward(self, value):
        self.adaptive_length_reward = value
