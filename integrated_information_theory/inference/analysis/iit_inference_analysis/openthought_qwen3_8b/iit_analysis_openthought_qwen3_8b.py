from integrated_information_theory.inference.analysis.iit_inference_analysis.iit_inference_analysis import iit_inference_analysis 
from integrated_information_theory.datasets.math.open_thoughts_dataset import open_thoughts_dataset
from integrated_information_theory.datasets.dataset_config import dataset_config
from integrated_information_theory.logger.inference.iit_inference_analysis.iit_inference_analysis_logger import iit_inference_analysis_logger

class iit_analysis_openthought_qwen3_8b(iit_inference_analysis): 

    def __init__(self, modelname: str, num_sequences: int, num_sequences_tpm: int) -> None:
        super().__init__(modelname, num_sequences, num_sequences_tpm)
        

    def get_dataset(self) -> open_thoughts_dataset:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_max_test_dataset_size(30)
            self.dataset = open_thoughts_dataset(config)
        return self.dataset

    def get_max_new_tokens(self) -> int:
        return 5000

    def create_logger(self, run_number) -> iit_inference_analysis_logger:
        return iit_inference_analysis_logger(log_file_name = f'integrated_information_theory/inference/analysis/iit_inference_analysis/openthought_qwen3_8b/run_{run_number}/iit_analysis_openthought_qwen3_8b.csv')


t = iit_analysis_openthought_qwen3_8b(modelname='Qwen/Qwen3-8B', num_sequences=192, num_sequences_tpm=4)
# t.run(from_run_number=2, to_run_number=3)
t.analyze(from_run_number=2, to_run_number=3)

