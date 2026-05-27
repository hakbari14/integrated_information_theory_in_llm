from integrated_information_theory.enums_class import tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum, iit_layer_type_enum, iit_threashold_type_enum
from abc import ABC, abstractmethod


class integrated_information_theory_config(ABC): 

    def __init__(self):
        self.adaptive_dim = None
        self.is_fixed_coefficient = None
        self.coefficient_mean_reward_dimension = 0.5
        self.coefficient_std_reward_dimension = 0.5
        self.reduced_dim = None
        self.tpm_creation_type = None
        self.layer_type = None
        self.threashold_type = None
        self.name = None

        self.last_layer_computation_type = None
        self.last_layer_computation_param = None

        self.granularity = None
        self.chunk_size = None
    
    def validate(self): 
        if self.get_adaptive_dim() is None:
            raise Exception('adaptive dimension is required')
        if self.get_adaptive_dim() == False and self.get_reduced_dim() is None:
            raise Exception('reduced dimension is required')
        if self.get_adaptive_dim() == True and self.get_reduced_dim() is not None:
            raise Exception('the number of dimensions is determined automatically in adaptive mode')
        if self.get_adaptive_dim() == True and self.get_is_fixed_coefficient() is None:
            raise Exception('is fixed coefficient is required')
        if self.get_is_fixed_coefficient() == True and self.get_coefficient_mean_reward_dimension() is None:
            raise Exception('coefficient_mean_reward_dimension is required')
        if self.get_is_fixed_coefficient() == True and self.get_coefficient_std_reward_dimension() is None:
            raise Exception('coefficient_std_reward_dimension is required')

        if self.get_tpm_creation_type() is None:
            raise Exception('TPM creation type is required')
        if tpm_creation_type_enum.TRAJECTORY != self.get_tpm_creation_type() and tpm_creation_type_enum.PROMPT != self.get_tpm_creation_type() and tpm_creation_type_enum.BATCH != self.get_tpm_creation_type():
            raise Exception('TPM creation type has not been correctly determined')

        if self.get_layer_type() is None:
            raise Exception('Layer type is required')
        if iit_layer_type_enum.ALL != self.get_layer_type() and iit_layer_type_enum.LAST != self.get_layer_type() and iit_layer_type_enum.SOME != self.get_layer_type():
            raise Exception('Layer type has not been correctly determined')

        if self.get_threashold_type() is None:
            raise Exception('Threashold type is required')
        if iit_threashold_type_enum.AVERAGE != self.get_threashold_type() and iit_threashold_type_enum.MEDIAN != self.get_threashold_type():
            raise Exception('Threashold type has not been correctly determined')

        if last_layer_computation_type_enum.TANH != self.get_last_layer_computation_type() and last_layer_computation_type_enum.EXP != self.get_last_layer_computation_type() and last_layer_computation_type_enum.IDENTITY != self.get_last_layer_computation_type():
            raise Exception('last layer computation type has not been correctly determined')

        if self.get_last_layer_computation_param() is None:
            raise Exception('last layer computation param is required')

        if granularity_enum.CHUNK != self.get_granularity() and granularity_enum.TOKEN != self.get_granularity():
            raise Exception('granularity has not been correctly determined')

        if granularity_enum.CHUNK == self.get_granularity() and self.get_chunk_size() is None:
            raise Exception('chunk_size is required')
        

    def get_adaptive_dim(self): 
        return self.adaptive_dim

    def set_adaptive_dim(self, value): 
        self.adaptive_dim = value

    def get_is_fixed_coefficient(self): 
        return self.is_fixed_coefficient

    def set_is_fixed_coefficient(self, value): 
        self.is_fixed_coefficient = value

    def get_coefficient_mean_reward_dimension(self): 
        return self.coefficient_mean_reward_dimension

    def set_coefficient_mean_reward_dimension(self, value): 
        self.coefficient_mean_reward_dimension = value

    def get_coefficient_std_reward_dimension(self): 
        return self.coefficient_std_reward_dimension

    def set_coefficient_std_reward_dimension(self, value): 
        self.coefficient_std_reward_dimension = value

    def get_reduced_dim(self): 
        return self.reduced_dim

    def set_reduced_dim(self, value): 
        self.reduced_dim = value

    def get_tpm_creation_type(self): 
        return self.tpm_creation_type

    def set_tpm_creation_type(self, value): 
        self.tpm_creation_type = value

    def get_layer_type(self): 
        return self.layer_type

    def set_layer_type(self, value): 
        self.layer_type = value

    def get_threashold_type(self): 
        return self.threashold_type

    def set_threashold_type(self, value): 
        self.threashold_type = value

    def get_last_layer_computation_type(self): 
        return self.last_layer_computation_type

    def set_last_layer_computation_type(self, value): 
        self.last_layer_computation_type = value
    
    def get_last_layer_computation_param(self): 
        return self.last_layer_computation_param

    def set_last_layer_computation_param(self, value): 
        self.last_layer_computation_param = value
    
    def get_granularity(self): 
        return self.granularity

    def set_granularity(self, value): 
        self.granularity = value
    
    def get_chunk_size(self): 
        return self.chunk_size

    def set_chunk_size(self, value): 
        self.chunk_size = value

    def get_name(self): 
        return self.name

    def set_name(self, value): 
        self.name = value
    
