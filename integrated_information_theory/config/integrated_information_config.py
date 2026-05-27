from integrated_information_theory.config.integrated_information_theory_config import integrated_information_theory_config
from integrated_information_theory.enums_class import ii_phi_type_enum

class integrated_information_config(integrated_information_theory_config): 

    def __init__(self):
        super().__init__()
        self.phi_type = None
    
    def vaidate(self):
        super().vaidate()

        if self.get_phi_type() is None:
            raise Exception('phi type is required')
        if ii_phi_type_enum.SYSTEM_PHI != self.get_phi_type() and ii_phi_type_enum.BIG_PHI != self.get_phi_type():
            raise Exception('phi type has not been correctly determined')

    def get_phi_type(self): 
        return self.phi_type

    def set_phi_type(self, value): 
        self.phi_type = value
    
