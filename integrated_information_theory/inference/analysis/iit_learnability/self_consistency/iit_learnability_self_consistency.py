from abc import ABC, abstractmethod
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from tqdm import tqdm
from vllm import LLM, SamplingParams
from integrated_information_theory.integrated_information_theory import integrated_information_theory
from integrated_information_theory.inference.analysis.iit_learnability.self_consistency.iit_learnability_entity import iit_learnability_entity 
from integrated_information_theory.inference.analysis.iit_learnability.self_consistency.iit_learnability_detail_entity import iit_learnability_detail_entity 
from integrated_information_theory.logger.inference.iit_learnability.iit_learnability_log_entity import iit_learnability_log_entity
from integrated_information_theory.logger.inference.iit_learnability.iit_learnability_logger import iit_learnability_logger
from integrated_information_theory.datasets.dataset_handler import dataset_handler
from integrated_information_theory.llm_representation import llm_representation
from integrated_information_theory.enums_class import iit_layer_type_enum

import torch
import gc
import logging

logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)        


class iit_learnability_self_consistency(ABC): 

    def __init__(self, modelname: str, num_sequences: int):
        self.modelname = modelname
        self.num_sequences = num_sequences
        
        if self.modelname is None:
            raise Exception('modelname is required')
        if self.num_sequences is None:
            raise Exception('num_sequences is required')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.representation = llm_representation()
        self.iit_calculator_list = None
        self.dataset = None


    def run(self, from_run_number: int , to_run_number: int) -> None:
        print(f"{'*' * 100}  {self.modelname}  {'*' * 100}")
        for run_number in range(from_run_number,to_run_number):
            print(f"{'*' * 100}  Run Number {run_number}  {'*' * 100}")
            
            entity_list: list[iit_learnability_entity] = self.generate_self_consistency(run_number=run_number)
            log_list: list[iit_learnability_log_entity] = self.calculate_iit_embedding(entity_list)
            
            logger = self.create_logger(run_number)
            logger.add_to_buffer_list(log_list)
            logger.write_to_log_file()
            
            print(f"{'*' * 210}")

    @torch.inference_mode()
    def generate_self_consistency(self, run_number) -> list[iit_learnability_entity]: 
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
        
        entity_list: list[iit_learnability_entity] = []
        for i in tqdm(range(0, len(test_dataset)), desc="Processing", unit="step"):
            x = test_dataset[i]
            
            entity = iit_learnability_entity()
            entity.ID = i
            entity.sample_ID = x['sample_id']
            entity.problem_id = x['problem_id']
            entity.split = x['split']
            entity.question = x['question']
            entity.prompt = x['prompt']
            entity.target = x['target']
            
            try:
                prompt_list = []
                prompt_list.append(entity.prompt)
                outputs = model.generate(prompt_list, sampling_params, use_tqdm=False)
                for j, output in enumerate(outputs[0].outputs):
                    idx = i + j

                    detail_entity = iit_learnability_detail_entity()
                    detail_entity.index = idx
                    detail_entity.completion = output.text
                    detail_entity.token_count = len(output.token_ids)
                    
                    try:
                        final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(entity.prompt, str(entity.completion), entity.target)
                        if final_answer is None or compared_final_answer is None: continue
                        detail_entity.final_answer = final_answer
                        detail_entity.compared_final_answer = compared_final_answer
                        detail_entity.accuracy = accuracy
                    except Exception as e:
                        logging.exception("An exception occurred")                        
                        print(f"[WARN]: {e}")
                                        
                    entity.add_consistency_list(detail_entity)
                    
                entity_list.append(entity)
            except Exception as e:
                logging.exception("An exception occurred")                        
                print(f"[WARN]: {e}")
            
       
        self.released_gpu_memory(model, tokenizer)
        return entity_list

    @torch.inference_mode()
    def calculate_iit_embedding(self, entity_list: list[iit_learnability_entity]) -> list[iit_learnability_log_entity]: 
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

        log_list: list[iit_learnability_log_entity] = []
        for entity in tqdm(entity_list, desc="Calculate IIT Embedding", unit="step"): 
            refine_prompt = self.representation.clean_prompt_for_phi(entity.prompt)
            prompt_emb, _, _ = self.representation.optimize_extract_representation(refine_prompt, model, tokenizer, iit_layer_type_enum.SOME)
            
            for detail_entity in entity.consistency_list:     
                log = self.create_log_entity(entity, detail_entity)
    
                try:
                    if detail_entity.completion is not None:
                        completion_emb, _, _ = self.representation.optimize_extract_representation(detail_entity.get_completion(), model, tokenizer, iit_layer_type_enum.SOME)
                        log.completion_mean_representation = completion_emb.mean(axis=1)
                        
                        response_for_pca = integrated_information_theory.refine_prompt_response_embedding(prompt_emb, completion_emb)
                    
                        log.iit_completion_embedding_3 = integrated_information_theory.reduce_prompt_response_embedding(response_for_pca, dimension = 3)
                        log.iit_completion_embedding_4 = integrated_information_theory.reduce_prompt_response_embedding(response_for_pca, dimension = 4)
                        log.iit_completion_embedding_5 = integrated_information_theory.reduce_prompt_response_embedding(response_for_pca, dimension = 5)
                except Exception as e:
                    print(f"[WARN] Load Embedding: {e}")
                    
                log_list.append(log)       
                
        self.released_gpu_memory(model, tokenizer)
        return log_list

    def create_log_entity(self, entity: iit_learnability_entity, detail_entity: iit_learnability_detail_entity) -> iit_learnability_log_entity:
        log = iit_learnability_log_entity()
        
        log.ID = f'{entity.ID}_{detail_entity.index}'
        log.sample_ID = entity.sample_ID
        log.problem_id = entity.problem_id
        log.split = entity.split
        log.prompt = entity.prompt
        log.target = entity.target
        log.question = entity.question
        
        log.completion = detail_entity.completion
        log.final_answer = detail_entity.final_answer
        log.compared_final_answer = detail_entity.compared_final_answer
        log.token_count = detail_entity.token_count
        log.accuracy = detail_entity.accuracy
        return log
        

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
    def create_logger(self, run_number) -> iit_learnability_logger:
        pass

