from integrated_information_theory.config.integrated_information_theory_config import integrated_information_theory_config
from integrated_information_theory.enums_class import ii_calculation_type_enum

class intrinsic_information_config(integrated_information_theory_config): 

    def __init__(self):
        super().__init__()
        self.ii_type = None
        self.calculation_type = None
        self.has_informativeness = None
        self.has_selectivity = None

    def vaidate(self):
        super().vaidate()

        if self.get_calculation_type() is None:
            raise Exception('calculation type is required')
        if ii_calculation_type_enum.SUM != self.get_calculation_type() and ii_calculation_type_enum.MAX != self.get_calculation_type():
            raise Exception('calculation type has not been correctly determined')

    def get_calculation_type(self): 
        return self.calculation_type

    def set_calculation_type(self, value): 
        self.calculation_type = value
    
    def get_has_selectivity(self): 
        return self.has_selectivity

    def set_has_selectivity(self, value): 
        self.has_selectivity = value

    def get_has_informativeness(self): 
        return self.has_informativeness

    def set_has_informativeness(self, value): 
        self.has_informativeness = value
    
    
