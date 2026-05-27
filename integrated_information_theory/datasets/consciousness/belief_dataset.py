from integrated_information_theory.datasets.consciousness.consciousness_dataset import consciousness_dataset

class belief_dataset(consciousness_dataset): 

    def __init__(self, config):
        super().__init__(config)

    def get_dataset_path(self):
        return "/home/hr_akbari/research/LLM_PostTraining/integrated_information_theory/datasets/data/consciousness/belief.json"

