from abc import ABC, abstractmethod
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from tqdm import tqdm
from vllm import LLM, SamplingParams

from integrated_information_theory.datasets.dataset_handler import dataset_handler
from integrated_information_theory.llm_representation import llm_representation
from integrated_information_theory.inference.analysis.iit_inference_analysis.iit_inference_analysis_entity import iit_inference_analysis_entity
from integrated_information_theory.inference.analysis.iit_inference_analysis.iit_inference_analysis_self_consistency_entity import iit_inference_analysis_self_consistency_entity
from integrated_information_theory.logger.inference.iit_inference_analysis.iit_inference_analysis_log_entity import iit_inference_analysis_log_entity
from integrated_information_theory.logger.inference.iit_inference_analysis.iit_inference_analysis_logger import iit_inference_analysis_logger
from integrated_information_theory.entity.iit_entity import iit_entity
from integrated_information_theory.integrated_information_theory import integrated_information_theory
from integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from integrated_information_theory.config.integrated_information_config import integrated_information_config
from integrated_information_theory.enums_class import ii_calculation_type_enum, tpm_creation_type_enum, last_layer_computation_type_enum, iit_layer_type_enum, iit_threashold_type_enum,ii_phi_type_enum, granularity_enum
from integrated_information_theory.intrinsic_information import intrinsic_information
from integrated_information_theory.integrated_information import integrated_information
from integrated_information_theory.inference.analysis.iit_inference_analysis.iit_reward_analyzer import iit_reward_analyzer

import pandas as pd
import torch
import gc
import numpy as np 
import logging

logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)        


class iit_inference_analysis(ABC): 

    def __init__(self, modelname: str, num_sequences: int, num_sequences_tpm: int):
        self.modelname = modelname
        self.num_sequences = num_sequences
        self.num_sequences_tpm = num_sequences_tpm
        
        if self.modelname is None:
            raise Exception('modelname is required')
        if self.num_sequences is None:
            raise Exception('num_sequences is required')
        if self.num_sequences_tpm is None:
            raise Exception('num_sequences_tpm is required')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.representation = llm_representation()
        self.iit_calculator_list = None
        self.dataset = None


    def run(self, from_run_number: int , to_run_number: int) -> None:
        print(f"{'*' * 100}  {self.modelname}  {'*' * 100}")
        for run_number in range(from_run_number,to_run_number):
            print(f"{'*' * 100}  Run Number {run_number}  {'*' * 100}")
            
            log_list: list[iit_inference_analysis_entity] = self.generate_self_consistency(run_number=run_number)
            log_analysis_list: list[iit_inference_analysis_log_entity] = self.calculate_iit(log_list=log_list)
            
            logger = self.create_logger(run_number)
            logger.add_to_buffer_list(log_analysis_list)
            logger.write_to_log_file()
            
            print(f"{'*' * 210}")

    def best_of_n_analysis(self, from_run_number: int , to_run_number: int, num_subsets=1000) -> None:
        print(f"{'*' * 100}  {self.modelname}  {'*' * 100}")
        for run_number in range(from_run_number,to_run_number):
            print(f"{'*' * 100}  Run Number {run_number}  {'*' * 100}")
            
            logger = self.create_logger(run_number)
            df = pd.read_csv(logger.get_log_file_name())
            analyzer = iit_reward_analyzer(dataframe = df, num_subsets = num_subsets)
            analyzer.test()
            
            print(f"{'*' * 210}")

    def variance_by_prompt_analysis(self, from_run_number: int , to_run_number: int, num_subsets=1000) -> None:
        print(f"{'*' * 100}  {self.modelname}  {'*' * 100}")
        for run_number in range(from_run_number,to_run_number):
            print(f"{'*' * 100}  Run Number {run_number}  {'*' * 100}")
            
            logger = self.create_logger(run_number)
            df = pd.read_csv(logger.get_log_file_name())
            analyzer = iit_reward_analyzer(dataframe = df, num_subsets = num_subsets)
            analyzer.reward_variance_by_prompt()
            
            print(f"{'*' * 210}")

    @torch.inference_mode()
    def calculate_iit(self, log_list: list[iit_inference_analysis_entity]) -> list[iit_inference_analysis_log_entity]: 
        bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = getattr(torch, "bfloat16"),
        bnb_4bit_use_double_quant = False,
        )
        model = AutoModelForCausalLM.from_pretrained(self.modelname, quantization_config = bnb_config)
        model.config.use_cache = False
        model.config.pretraining_tp = 1        
        tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        print(f"{'*' * 90}  Calculate IIT {'*' * 90}")
        
        log_analysis_list: list[iit_inference_analysis_log_entity] = []
        for log in tqdm(log_list, desc="Integrated Information Processing", unit="step"): 
            try:
                calculated_list = self.load_embedding(log, model, tokenizer)
                
                origin_entity_list: list[iit_entity] = []
                for entity, log_analysis in calculated_list:
                    origin_entity_list.append(entity)
                
                if len(origin_entity_list) == 0: continue
                 
                for iit_calculator in self.get_iit_calculator_list():
                    try: 
                        entity_list = iit_entity.clone_list(origin_entity_list)
                        entity_list: list[iit_entity] = iit_calculator.calculate(entity_list)
                        
                        for entity, log_analysis in calculated_list:
                            filter_founded_entity: iit_entity = list(filter(lambda x: x.key == entity.key, entity_list))
                            if len(filter_founded_entity) == 0 : continue

                            founded_entity = filter_founded_entity[0]
                            if iit_calculator.get_config().get_name() == 'IIR':
                                log_analysis.completion_embedding_shape_iir = founded_entity.get_completion_embedding_shape()
                                log_analysis.iir_reward = founded_entity.get_iit_reward()
                                log_analysis.iir_reward_raw = founded_entity.get_iit_reward_raw()
                                log_analysis.iir_reward_raw_actual = founded_entity.get_iit_reward_raw_actual()
                            elif iit_calculator.get_config().get_name() == 'Phi_S':
                                log_analysis.completion_embedding_shape_phi_s = founded_entity.get_completion_embedding_shape()
                                log_analysis.phi_s_reward = founded_entity.get_iit_reward()
                                log_analysis.phi_s_reward_raw = founded_entity.get_iit_reward_raw()
                                log_analysis.phi_s_reward_raw_actual = founded_entity.get_iit_reward_raw_actual()
                            elif iit_calculator.get_config().get_name() == 'Phi':
                                log_analysis.completion_embedding_shape_phi = founded_entity.get_completion_embedding_shape()
                                log_analysis.phi_reward = founded_entity.get_iit_reward()
                                log_analysis.phi_reward_raw = founded_entity.get_iit_reward_raw()
                                log_analysis.phi_reward_raw_actual = founded_entity.get_iit_reward_raw_actual()
                    
                        del entity_list
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception as e:
                        logging.exception("An exception occurred")                        
                        print(f"[WARN] generate failed: {e}")

                del origin_entity_list
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                logging.exception("An exception occurred")                        
                print(f"[WARN] generate failed: {e}")
            
            for entity, log_analysis in calculated_list:
                log_analysis_list.append(log_analysis)
                del entity
            gc.collect()
            torch.cuda.empty_cache()
                
        self.released_gpu_memory(model, tokenizer)
        return log_analysis_list

   
    @torch.inference_mode()
    def generate_self_consistency(self, run_number) -> list[iit_inference_analysis_entity]: 
        model = LLM(model=self.modelname, tensor_parallel_size=1, trust_remote_code=True,)
        tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        print(f"{'*' * 90}  Generate Self Consistency Run Number {run_number} {'*' * 90}")

        _, test_dataset = self.get_dataset().preprocess_dataset()
        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=1.0, 
                n = self.num_sequences, 
                top_p= 0.9, 
                top_k=50
            )
        
        log_list: list[iit_inference_analysis_entity] = []
        for i in tqdm(range(0, len(test_dataset)), desc="Processing", unit="step"):
            x = test_dataset[i]
            
            log_detail_list: list[iit_inference_analysis_self_consistency_entity] = []
            try:
                prompt = x['prompt']
                target = x['target']
                prompt_list = []
                prompt_list.append(prompt)
                outputs = model.generate(prompt_list, sampling_params, use_tqdm=False)
                for j, output in enumerate(outputs[0].outputs):
                    idx = i + j

                    log_detail = iit_inference_analysis_self_consistency_entity()
                    log_detail.index = idx
                    log_detail.completion = output.text
                    log_detail.token_count = len(output.token_ids)
                    
                    try:
                        final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, str(log_detail.completion), target)
                        if final_answer is None or compared_final_answer is None: continue
                        log_detail.final_answer = final_answer
                        log_detail.compared_final_answer = compared_final_answer
                        log_detail.accuracy = accuracy
                    except Exception as e:
                        logging.exception("An exception occurred")                        
                        print(f"[WARN]: {e}")
                                        
                    log_detail_list.append(log_detail)
            except Exception as e:
                logging.exception("An exception occurred")                        
                print(f"[WARN]: {e}")
            
            for i in range(0, len(log_detail_list), self.num_sequences_tpm):
                batch = log_detail_list[i:i + self.num_sequences_tpm]
                log = iit_inference_analysis_entity()
                log.ID = i
                log.sample_ID = x['sample_id']
                log.problem_id = x['problem_id']
                log.split = x['split']
                log.question = x['question']
                log.prompt = x['prompt']
                log.target = x['target']

                for j, entity in enumerate(batch):
                    log.add_consistency_list(entity)
                
                log_list.append(log)
       
        self.released_gpu_memory(model, tokenizer)
        return log_list

    @torch.inference_mode()
    def load_embedding(self, log: iit_inference_analysis_entity, model, tokenizer) -> list[tuple[iit_entity, iit_inference_analysis_log_entity]]: 
        calculated_list = []
        refine_prompt = self.representation.clean_prompt_for_phi(log.prompt)
        prompt_emb, _, _ = self.representation.extract_representation(refine_prompt, model, tokenizer, iit_layer_type_enum.SOME)
        
        for log_detail in log.consistency_list:     
            try:
                log_analysis = self.create_log_analysis_entity(log, log_detail)
                
                entity = iit_entity(key=log_detail.index)
                entity.set_promptID(log.sample_ID)
                entity.set_prompt(log.prompt)
                entity.set_prompt_embedding(prompt_emb)
                entity.set_completion(log_detail.completion)
                if log_detail.completion is not None:
                    completion_emb, _, _ = self.representation.extract_representation(entity.get_completion(), model, tokenizer, iit_layer_type_enum.SOME)
                    entity.set_completion_embedding_and_shape(completion_emb)
                    entity.set_token_count(completion_emb.shape[1])
                
                if entity.is_calcutable():
                    calculated_list.append((entity, log_analysis))

            except Exception as e:
                print(f"[WARN] Load Embedding: {e}")
       
        return calculated_list

    def create_log_analysis_entity(self, log: iit_inference_analysis_entity, log_detail: iit_inference_analysis_self_consistency_entity) -> iit_inference_analysis_log_entity:
        log_analysis = iit_inference_analysis_log_entity()
        
        log_analysis.ID = f'{log.ID}_{log_detail.index}'
        log_analysis.sample_ID = log.sample_ID
        log_analysis.problem_id = log.problem_id
        log_analysis.split = log.split
        log_analysis.prompt = log.prompt
        log_analysis.target = log.target
        log_analysis.question = log.question
        
        log_analysis.completion = log_detail.completion
        log_analysis.final_answer = log_detail.final_answer
        log_analysis.compared_final_answer = log_detail.compared_final_answer
        log_analysis.token_count = log_detail.token_count
        log_analysis.accuracy = log_detail.accuracy
        return log_analysis
        
    def get_iit_calculator_list(self) -> list[integrated_information_theory]:
        if self.iit_calculator_list is None: 
            iit_calculator_list: list[integrated_information_theory] = []

            config = intrinsic_information_config()
            config.set_name('IIR')
            config.set_adaptive_dim(False)
            config.set_calculation_type(ii_calculation_type_enum.SUM)
            config.set_reduced_dim(5)
            config.set_granularity(granularity_enum.TOKEN)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            iit_calculator_list.append(intrinsic_information(config)) 

            config = integrated_information_config()
            config.set_name('Phi_S') 
            config.set_adaptive_dim(False)
            config.set_phi_type(ii_phi_type_enum.SYSTEM_PHI)
            config.set_reduced_dim(4)
            config.set_granularity(granularity_enum.TOKEN)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            iit_calculator_list.append(integrated_information(config)) 

            config = integrated_information_config()
            config.set_name('Phi')
            config.set_adaptive_dim(False)
            config.set_phi_type(ii_phi_type_enum.BIG_PHI)
            config.set_reduced_dim(4)
            config.set_granularity(granularity_enum.TOKEN)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            iit_calculator_list.append(integrated_information(config)) 
            
            self.iit_calculator_list = iit_calculator_list
        
        return self.iit_calculator_list

    def released_gpu_memory(self, model, tokenizer) -> None:
        del model
        del tokenizer

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
        
    @abstractmethod
    def get_max_new_tokens(self) -> int:
        pass

    @abstractmethod
    def get_dataset(self) -> dataset_handler:
        pass

    @abstractmethod
    def create_logger(self, run_number) -> iit_inference_analysis_logger:
        pass

