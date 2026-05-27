import numpy as np

class dimension_entity:
    
    def __init__(self, dimension, iit_list):
        self.dimension = dimension
        self.iit_list = iit_list
        self.mean_iit_reward = None
        self.normalized_mean_iit_reward = None
        self.std_iit_reward = None
        self.normalized_std_iit_reward = None
        self.iit_reward_list = None

    def get_dimension(self):
        return self.dimension

    def set_dimension(self, value):
        self.dimension = value

    def get_iit_list(self):
        return self.iit_list

    def set_iit_list(self, value):
        self.iit_list = value

    def get_mean_iit_reward(self):
        return self.mean_iit_reward

    def set_mean_iit_reward(self, value):
        self.mean_iit_reward = value

    def get_normalized_mean_iit_reward(self):
        return self.normalized_mean_iit_reward

    def set_normalized_mean_iit_reward(self, value):
        self.normalized_mean_iit_reward = value

    def get_std_iit_reward(self):
        return self.std_iit_reward

    def set_std_iit_reward(self, value):
        self.std_iit_reward = value

    def get_normalized_std_iit_reward(self):
        return self.normalized_std_iit_reward

    def set_normalized_std_iit_reward(self, value):
        self.normalized_std_iit_reward = value

    def get_iit_reward_list(self):
        return self.iit_reward_list

    def set_iit_reward_list(self, value):
        self.iit_reward_list = value

    def get_ratio_metric(self):
        return self.get_normalized_mean_iit_reward() / (self.get_normalized_std_iit_reward() if self.get_normalized_std_iit_reward() != 0 else 1)

    def get_difference_metric(self, config):
        W_mean = config.get_coefficient_mean_reward_dimension()
        W_std = config.get_coefficient_std_reward_dimension()
        return W_mean * self.get_normalized_mean_iit_reward() - W_std* self.get_normalized_std_iit_reward()

    def set_mean_standard_deviation(self, iit_reward_list):
        self.set_iit_reward_list(iit_reward_list)
        if len(iit_reward_list) > 1: 
            self.set_mean_iit_reward(np.mean(iit_reward_list))
            self.set_std_iit_reward(np.std(iit_reward_list))
        elif len(iit_reward_list) == 1:
            self.set_mean_iit_reward(np.mean(iit_reward_list))
            self.set_std_iit_reward(0.0)
        else:
            self.set_mean_iit_reward(0.0)
            self.set_std_iit_reward(0.0)

    @staticmethod
    def z_score_normalization(dimension_entity_list): 
        mean_list = list(map(lambda x: x.get_mean_iit_reward(), dimension_entity_list))
        mean_mean = np.mean(mean_list)
        std_mean = np.std(mean_list) 
        if len(mean_list) == 1 or std_mean == 0 or std_mean is None:
            std_mean =  1

        std_list = list(map(lambda x: x.get_std_iit_reward(), dimension_entity_list))
        mean_std = np.mean(std_list)
        std_std = np.std(std_list)
        if len(std_list) == 1 or std_std == 0 or std_std is None:
            std_std =  1

        for entity in dimension_entity_list:
            entity.set_normalized_mean_iit_reward((entity.get_mean_iit_reward() - mean_mean) / std_mean)
            entity.set_normalized_std_iit_reward((entity.get_std_iit_reward() - mean_std) / std_std)
        
        return dimension_entity_list
