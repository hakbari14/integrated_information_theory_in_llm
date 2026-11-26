from integrated_information_theory.inference.analysis.iit_learnability.self_consistency.iit_learnability_self_consistency import iit_learnability_self_consistency
from integrated_information_theory.datasets.math.open_thoughts_dataset import open_thoughts_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.iit_learnability.iit_learnability_logger import iit_learnability_logger

class iit_learnability_openthought_llama_3_1_8b_Instruct(iit_learnability_self_consistency): 

    def __init__(self, modelname: str, num_sequences: int) -> None:
        super().__init__(modelname, num_sequences)
        

    def get_dataset(self) -> open_thoughts_dataset:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_max_test_dataset_size(500)
            self.dataset = open_thoughts_dataset(config)
        return self.dataset

    def get_max_new_tokens(self) -> int:
        return 5000

    def create_logger(self, run_number) -> iit_learnability_logger:
        return iit_learnability_logger(log_file_name = f'integrated_information_theory/inference/analysis/iit_learnability/self_consistency/openthought_llama_3_1_8b_Instruct/run_{run_number}/iit_learnability_openthought_llama_3_1_8b_Instruct.csv')


t = iit_learnability_openthought_llama_3_1_8b_Instruct(modelname='/home/hr_akbari/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659', num_sequences=192)
t.run(from_run_number=2, to_run_number=3)

