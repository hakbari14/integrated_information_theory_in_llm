from abc import ABC, abstractmethod
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging,)
from integrated_information_theory.logger.inference.accuracy.inference_accuracy_log_entity import inference_accuracy_log_entity
from integrated_information_theory.logger.inference.faithfulness.inference_faithfulness_log_entity import inference_faithfulness_log_entity
from integrated_information_theory.logger.log_token_entity import log_token_entity
from integrated_information_theory.logger.inference.self_consistency.self_consistency_log_entity import self_consistency_log_entity
from integrated_information_theory.logger.inference.self_consistency.self_consistency_log_detail_entity import self_consistency_log_detail_entity
from integrated_information_theory.logger.inference.reward_hacking.inference_reward_hacking_log_entity import inference_reward_hacking_log_entity
from integrated_information_theory.logger.inference.causalbench.inference_causalbench_log_entity import inference_causalbench_log_entity
from integrated_information_theory.logger.inference.code.inference_code_logger import inference_code_logger
from integrated_information_theory.logger.inference.code.inference_code_log_entity import inference_code_log_entity
from integrated_information_theory.entity.iit_entity import iit_entity
from integrated_information_theory.llm_representation import llm_representation
from integrated_information_theory.enums_class import llm_pipeline_type_enum, iit_layer_type_enum
from integrated_information_theory.utils import my_utils
from tqdm import tqdm
import torch
import gc
import pandas as pd
from peft import PeftModel
from vllm import LLM, SamplingParams
import jsonlines
# from human_eval.evaluate_functional_correctness import evaluate_functional_correctness
import numpy as np
from pathlib import Path


class integrated_information_inference(ABC): 

    def __init__(self, model_name, peft_checkpoint_path=None):
        self.model_name = model_name
        self.pipeline_type = llm_pipeline_type_enum.INFERENCE
        if self.model_name is None:
            raise Exception('model name is required')

        self.representation = llm_representation()
        self.dataset = None
        config = self.get_dataset().get_config()
        config.set_pipeline_type(llm_pipeline_type_enum.INFERENCE)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.iit_calculator = None
        self.logger = None

        self.peft_checkpoint_path = peft_checkpoint_path

    @torch.inference_mode()
    def calculate_accuracy(self): 
        train_dataset, test_dataset = self.get_dataset().preprocess_dataset()
        log_list = []
        print('Stage: Output generation')
        for idx, x in enumerate(tqdm(test_dataset)):
            prompt = x['prompt']
            sample_ID = x['sample_id']
            split = x['split']
            target = x['target']
            problem_id = x['problem_id']
            log = inference_accuracy_log_entity(idx, sample_ID, problem_id, split, prompt, target)

            try:
                foundational_outputs_sentence = self.generate(prompt)
                output = self.tokenizer.batch_decode(foundational_outputs_sentence, skip_special_tokens=True)
                if output is not None and len(output) == 1:
                    completion = output[0]
                    final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), target)
                    log.set_completion(completion)
                    log.set_final_answer(final_answer)
                    log.set_accuracy(accuracy)

            except Exception as e:
                print(f"[WARN] generate failed: {e}")

            log_list.append(log)

        try:
            log_list, result_list = self.load_embedding_and_loss(log_list)
            log_list = self.calculate_iit(log_list, result_list)
        except Exception as e:
            print(f"[WARN] calculate iit : {e}")

        self.get_logger().add_to_buffer_list(log_list)
        self.get_logger().write_to_log_file()

    @torch.inference_mode()
    def calculate_accuracy_vllm(self, batch_size = 128): 
        _, test_dataset = self.get_dataset().preprocess_dataset()
        log_list = []

        print('Stage: Output generation')
        model = LLM(model=self.model_name, tensor_parallel_size=1, trust_remote_code=True,)
        sampling_params = SamplingParams (
                max_tokens=self.get_max_new_tokens(), 
                temperature = 0.7, 
                top_p = 1.0, 
                top_k = 50, 
                repetition_penalty = 1.1, 
            )

        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing batches"):
            batch = test_dataset[i : i + batch_size]
            prompt_list = batch['prompt']
            sample_ID_list = batch['sample_id']
            split_list = batch['split']
            target_list = batch['target']
            problem_id_list = batch['problem_id']

            outputs = model.generate(prompt_list, sampling_params)
            for j, output in enumerate(outputs):
                try:
                    idx = i + j
                    prompt = prompt_list[j]
                    sample_ID = sample_ID_list[j]
                    split = split_list[j]
                    target = target_list[j]
                    problem_id = problem_id_list[j]
                    log = inference_accuracy_log_entity(idx, sample_ID, problem_id, split, prompt, target)
                    
                    if output.outputs is None or len(output.outputs) != 1: continue
                    response = output.outputs[0]
                    completion = response.text
                    log.set_completion(completion)
                    log.set_token_count(len(response.token_ids))
                    
                    final_answer, accuracy, _ = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), target)
                    log.set_final_answer(final_answer)
                    log.set_accuracy(accuracy)
                    log_list.append(log)

                except Exception as e:
                    print(f"[WARN] generate failed: {e}")

        self.get_logger().add_to_buffer_list(log_list)
        self.get_logger().write_to_log_file()

    def calculate_accuracy_consciousness(self, batch_size = 128): 
        _, test_dataset = self.get_dataset().preprocess_dataset()
        log_list = []
        print('Stage: Output generation')

        model = LLM(model=self.model_name, tensor_parallel_size=1, trust_remote_code=True,)
        sampling_params = SamplingParams (
                max_tokens=self.get_max_new_tokens(), 
                temperature = 1.0, 
                top_p = 1.0, 
                top_k = 50, 
                repetition_penalty = 1.1, 
            )

        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing batches"):
            batch = test_dataset[i : i + batch_size]
            prompt_list = batch['prompt']
            sample_ID_list = batch['sample_id']
            split_list = batch['split']
            target_list = batch['target']
            problem_id_list = batch['problem_id']

            outputs = model.generate(prompt_list, sampling_params)

            for j, output in enumerate(outputs):
                try:
                    idx = i + j
                    prompt = prompt_list[j]
                    sample_ID = sample_ID_list[j]
                    split = split_list[j]
                    target = target_list[j]
                    problem_id = problem_id_list[j]
                    log = inference_accuracy_log_entity(idx, sample_ID, problem_id, split, prompt, target)
                    
                    if output.outputs is None or len(output.outputs) != 1: continue
                    response = output.outputs[0]
                    completion = response.text
                    log.set_completion(completion)
                    log.set_token_count(len(response.token_ids))

                    final_answer, accuracy, _ = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), target)
                    log.set_final_answer(final_answer)
                    log.set_accuracy(accuracy)
                    log_list.append(log)

                except Exception as e:
                    print(f"[WARN] generate failed: {e}")
                
        self.get_logger().add_to_buffer_list(log_list)
        self.get_logger().write_to_log_file()
        

    def calculate_accuracy_causal_bench(self, batch_size = 128): 
        _, test_dataset = self.get_dataset().preprocess_dataset()
        log_list = []

        print('Stage: Output generation')
        model = LLM(model=self.model_name, tensor_parallel_size=1, trust_remote_code=True,)
        sampling_params = SamplingParams (
                max_tokens=self.get_max_new_tokens(), 
                temperature = 1.0, 
                top_p = 1.0, 
                top_k = 50, 
                repetition_penalty = 1.1, 
            )

        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing batches"):
            batch = test_dataset[i : i + batch_size]
            prompt_list = batch['prompt']
            sample_ID_list = batch['sample_id']
            split_list = batch['split']
            target_list = batch['target']
            problem_id_list = batch['problem_id']
            question_type_list = batch['question_type']
            scenario_ID_list = batch['scenario_ID']

            outputs = model.generate(prompt_list, sampling_params)
            for j, output in enumerate(outputs):
                try:
                    idx = i + j
                    prompt = prompt_list[j]
                    sample_ID = sample_ID_list[j]
                    split = split_list[j]
                    target = target_list[j]
                    problem_id = problem_id_list[j]
                    question_type = question_type_list[j]
                    scenario_ID = scenario_ID_list[j]
                    log = inference_causalbench_log_entity(idx, sample_ID, problem_id, split, prompt, target, question_type, scenario_ID)
                    
                    if output.outputs is None or len(output.outputs) != 1: continue
                    response = output.outputs[0]
                    completion = response.text
                    log.set_completion(completion)
                    log.set_token_count(len(response.token_ids))
                    
                    final_answer, accuracy, _ = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), target)
                    log.set_final_answer(final_answer)
                    log.set_accuracy(accuracy)
                    log_list.append(log)

                except Exception as e:
                    print(f"[WARN] generate failed: {e}")

        self.get_logger().add_to_buffer_list(log_list)
        self.get_logger().write_to_log_file()
        return None

    @torch.inference_mode()
    def calculate_accuracy_code(self, batch_size = 128): 
        _, test_dataset = self.get_dataset().preprocess_dataset()

        print('Stage: Output generation')
        model = LLM(model=self.model_name, tensor_parallel_size=1, trust_remote_code=True,)
        sampling_params = SamplingParams (
                max_tokens=self.get_max_new_tokens(), 
                temperature = 1.0, 
                top_p = 1.0, 
                top_k = 50, 
                repetition_penalty = 1.1, 
            )

        samples = []        
        log_list = []
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing batches"):
            batch = test_dataset[i : i + batch_size]
            prompt_list = batch['prompt']
            sample_ID_list = batch['sample_id']
            split_list = batch['split']
            target_list = batch['target']
            problem_id_list = batch['problem_id']
            entry_point_list = batch['entry_point']

            try:
                outputs = model.generate(prompt_list, sampling_params)

                for j, output in enumerate(outputs):
                    idx = i + j
                    prompt = prompt_list[j]
                    sample_ID = sample_ID_list[j]
                    split = split_list[j]
                    target = target_list[j]
                    problem_id = problem_id_list[j]
                    entry_point = entry_point_list[j]

                    log = inference_code_log_entity(idx, sample_ID, problem_id, split, prompt, target)
                    
                    if output.outputs is None or len(output.outputs) != 1: continue
                    response = output.outputs[0]
                    completion = response.text
                    log.set_completion(completion)
                    log.set_token_count(len(response.token_ids))

                    extracted_code = self.get_dataset().code_extraction(str(completion), entry_point)
                    log.set_final_answer(extracted_code)
                    log_list.append(log)

                    samples.append(dict(task_id=problem_id, completion=extracted_code))                    

            except Exception as e:
                print(f"[WARN] generate failed: {e}")

        with jsonlines.open(self.get_output_file_path_code(), 'w') as writer:
            writer.write_all(samples)
        print(f"Generated samples saved to {self.get_output_file_path_code()}")

        print("\nRunning official HumanEval evaluation...")
        # result = evaluate_functional_correctness(self.get_output_file_path_code())

        result_file = self.get_output_file_path_code() + '_results.jsonl'
        result_df = pd.read_json(result_file, lines=True)        
        for index, row in result_df.iterrows():
            item_log_list = list(filter(lambda x: x.get_problem_id() == result_df.loc[index, "task_id"] , log_list))
            if item_log_list is None or len(item_log_list) == 0: continue
            
            item_log_list[0].set_accuracy(result_df.loc[index, "passed"])

        self.get_logger().add_to_buffer_list(log_list)
        self.get_logger().write_to_log_file()

    @torch.inference_mode()
    def calculate_accuracy_faithfulness(self): 
        train_dataset, test_dataset = self.get_dataset().preprocess_dataset()
        log_list = []
        print('Stage: Output generation')
        for idx, x in enumerate(tqdm(test_dataset)):
            prompt_1 = x['prompt_1']
            prompt_2 = x['prompt_2']
            sample_ID = x['sample_id']
            split = x['split']
            answer = x['answer']
            question_by_qid = x['question_by_qid']
            problem_id = x['problem_id']
            log = inference_faithfulness_log_entity(idx, sample_ID, problem_id, split, prompt_1, answer, prompt_2, question_by_qid)

            try:
                foundational_outputs_sentence_1 = self.generate(prompt_1)
                output_1 = self.tokenizer.batch_decode(foundational_outputs_sentence_1, skip_special_tokens=True)

                foundational_outputs_sentence_2 = self.generate(prompt_2)
                output_2 = self.tokenizer.batch_decode(foundational_outputs_sentence_2, skip_special_tokens=True)

                if output_1 is not None and len(output_1) == 1 and output_2 is not None and len(output_2) == 1:
                    completion_1 = output_1[0]
                    final_answer_1 = self.get_dataset().final_answer_extraction(str(completion_1))
                    log.set_completion(completion_1)
                    log.set_final_answer(final_answer_1)

                    completion_2 = output_2[0]
                    final_answer_2 = self.get_dataset().final_answer_extraction(str(completion_2))
                    log.set_completion_2(completion_2)
                    log.set_final_answer_2(final_answer_2)
                    if final_answer_1 is not None or final_answer_2 is not None:
                        log.set_accuracy(final_answer_1 != final_answer_2)

            except Exception as e:
                print(f"[WARN] generate failed: {e}")

            log_list.append(log)

        self.get_logger().add_to_buffer_list(log_list)
        self.get_logger().write_to_log_file()

    @torch.inference_mode()
    def calculate_accuracy_self_consistency(self, temperature = 0.7, num_sequences = 10, top_p = 0.9, top_k = 50): #top_p:0.95, temprature:0.7, num_seq:5
        train_dataset, test_dataset = self.get_dataset().preprocess_dataset()
        log_list = []
        print('Stage: Output generation')
        for idx, x in enumerate(tqdm(test_dataset)):
            prompt = x['prompt']
            sample_ID = x['sample_id']
            split = x['split']
            target = x['target']
            problem_id = None #x['problem_id']
            log = self_consistency_log_entity(idx, sample_ID, problem_id, split, prompt, target)

            try:
                foundational_outputs_sentence = self.generate(prompt, temperature=temperature, num_return_sequences=num_sequences, top_p=top_p, top_k = top_k, do_sample=True)
                output = self.tokenizer.batch_decode(foundational_outputs_sentence, skip_special_tokens=True)
                for index in range(num_sequences):
                    completion = output[index]
                    final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), target)
                    log_detail = self_consistency_log_detail_entity()
                    log_detail.set_completion(completion)
                    log_detail.set_final_answer(final_answer)
                    log_detail.set_compared_final_answer(compared_final_answer)
                    log_detail.set_accuracy(accuracy)
                    log.add_consistency_list(log_detail)

            except Exception as e:
                print(f"[WARN] generate failed: {e}")

            vote_count = {}
            for log_detail in log.get_consistency_list():
                if log_detail.get_compared_final_answer() == None: continue
                if type(log_detail.get_compared_final_answer()) == str and len(log_detail.get_compared_final_answer()) == 0: continue
                vote_count[log_detail.get_compared_final_answer()] = vote_count.get(log_detail.get_compared_final_answer(), 0) + 1

            if len(vote_count.items()) > 0: 
                max_vote = max(vote_count.items(), key=lambda x: x[1])
                compared_final_answer = max_vote[0]
                log_detail_list = list(filter(lambda x: x.get_compared_final_answer() == compared_final_answer, log.get_consistency_list()))
                if len(log_detail_list) >= 1: 
                    log_detail = log_detail_list[0]
                    log.set_completion(log_detail.get_completion())
                    log.set_final_answer(log_detail.get_final_answer())
                    log.set_accuracy(log_detail.get_accuracy())
            else : 
                    if len(log.get_consistency_list()) > 0:
                        log_detail = log.get_consistency_list()[0]
                        log.set_completion(log_detail.get_completion())
                        log.set_final_answer(log_detail.get_final_answer())
                        log.set_accuracy(log_detail.get_accuracy())
            
            log_list.append(log)

        log_list, result_list = self.load_embedding_and_loss(log_list)
        self.get_logger().add_to_buffer_list(log_list)
        self.get_logger().write_to_log_file()


    @torch.inference_mode()
    def calculate_accuracy_self_consistency_vllm(self, batch_size = 12, temperature = 0.7, num_sequences = 10, top_p = 0.9, top_k = 50): #top_p:0.95, temprature:0.7, num_seq:5
        train_dataset, test_dataset = self.get_dataset().preprocess_dataset()
        log_list = []
        print('Stage: Output generation')
        model = LLM(model=self.model_name, tensor_parallel_size=1, trust_remote_code=True,)
        sampling_params = SamplingParams(max_tokens=self.get_max_new_tokens(), temperature=temperature, n = num_sequences, top_p= top_p, top_k=top_k)

        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing batches"):
            batch = test_dataset[i : i + batch_size]
            prompt_list = batch['prompt']
            sample_ID_list = batch['sample_id']
            split_list = batch['split']
            target_list = batch['target']
            problem_id_list = batch['problem_id']

            try:
                outputs = model.generate(prompt_list, sampling_params)

                for j, output in enumerate(outputs):
                    idx = i + j
                    prompt = prompt_list[j]
                    sample_ID = sample_ID_list[j]
                    split = split_list[j]
                    target = target_list[j]
                    problem_id = problem_id_list[j]
                    log = self_consistency_log_entity(idx, sample_ID, problem_id, split, prompt, target)
                    
                    if output.outputs is None: continue
                    for index in range(num_sequences):
                        response = output.outputs[index]
                        completion = response.text
                        
                        log_detail = self_consistency_log_detail_entity(f'{idx}_{index}')
                        log_detail.set_completion(completion)
                        log_detail.set_token_count(len(response.token_ids))

                        try:
                            final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), target)
                            log_detail.set_final_answer(final_answer)
                            log_detail.set_compared_final_answer(compared_final_answer)
                            log_detail.set_accuracy(accuracy)
                        except Exception as e:
                            print(f"[WARN] generate failed: {e}")
                            
                        log.add_consistency_list(log_detail)


                    vote_count = {}
                    for log_detail in log.get_consistency_list():
                        if log_detail.get_compared_final_answer() == None: continue
                        if type(log_detail.get_compared_final_answer()) == str and len(log_detail.get_compared_final_answer()) == 0: continue
                        vote_count[log_detail.get_compared_final_answer()] = vote_count.get(log_detail.get_compared_final_answer(), 0) + 1

                    if len(vote_count.items()) > 0: 
                        max_vote = max(vote_count.items(), key=lambda x: x[1])
                        compared_final_answer = max_vote[0]
                        log_detail_list = list(filter(lambda x: x.get_compared_final_answer() == compared_final_answer, log.get_consistency_list()))
                        if len(log_detail_list) >= 1: 
                            log_detail = log_detail_list[0]
                            log.set_completion(log_detail.get_completion())
                            log.set_final_answer(log_detail.get_final_answer())
                            log.set_accuracy(log_detail.get_accuracy())
                            log.set_token_count(log_detail.get_token_count())
                    else : 
                            if len(log.get_consistency_list()) > 0:
                                log_detail = log.get_consistency_list()[0]
                                log.set_completion(log_detail.get_completion())
                                log.set_final_answer(log_detail.get_final_answer())
                                log.set_accuracy(log_detail.get_accuracy())
                                log.set_token_count(log_detail.get_token_count())

                    log_list.append(log)

            except Exception as e:
                print(f"[WARN] generate failed: {e}")

        self.get_logger().add_to_buffer_list(log_list)
        self.get_logger().write_to_log_file()

    @torch.inference_mode()
    def calculate_accuracy_self_consistency_code(self, batch_size = 12, temperature = 0.7, num_sequences = 10, top_p = 0.9, top_k = 50, pass_at_k=3): 
        _, test_dataset = self.get_dataset().preprocess_dataset()
        log_list = []
        print('Stage: Output generation')
        model = LLM(model=self.model_name, tensor_parallel_size=1, trust_remote_code=True,)
        sampling_params = SamplingParams(max_tokens=self.get_max_new_tokens(), temperature=temperature, n = num_sequences, top_p= top_p, top_k=top_k)
        samples = []
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing batches"):
            batch = test_dataset[i : i + batch_size]
            prompt_list = batch['prompt']
            sample_ID_list = batch['sample_id']
            split_list = batch['split']
            target_list = batch['target']
            problem_id_list = batch['problem_id']
            entry_point_list = batch['entry_point']

            try:
                outputs = model.generate(prompt_list, sampling_params)

                for j, output in enumerate(outputs):
                    idx = i + j
                    prompt = prompt_list[j]
                    sample_ID = sample_ID_list[j]
                    split = split_list[j]
                    target = target_list[j]
                    problem_id = problem_id_list[j]
                    entry_point = entry_point_list[j]
                    log = self_consistency_log_entity(idx, sample_ID, problem_id, split, prompt, target)
                    
                    if output.outputs is None: continue
                    for index in range(num_sequences):
                        response = output.outputs[index]
                        completion = response.text

                        log_detail = self_consistency_log_detail_entity()
                        log_detail.set_completion(completion)
                        log_detail.set_token_count(len(response.token_ids))

                        try:
                            extracted_code = self.get_dataset().code_extraction(str(completion), entry_point)
                            log_detail.set_final_answer(extracted_code)
                            samples.append(dict(task_id=problem_id, completion=extracted_code))                    
                        except Exception as e:
                            print(f"[WARN] generate failed: {e}")
                            
                        log.add_consistency_list(log_detail)


                    log_list.append(log)

            except Exception as e:
                print(f"[WARN] generate failed: {e}")

        with jsonlines.open(self.get_output_file_path_code(), 'w') as writer:
            writer.write_all(samples)
        print(f"Generated samples saved to {self.get_output_file_path_code()}")

        print("\nRunning official HumanEval evaluation...")
        result = evaluate_functional_correctness(self.get_output_file_path_code())

        result_file = self.get_output_file_path_code() + '_results.jsonl'
        result_df = pd.read_json(result_file, lines=True)        
        for index, row in result_df.iterrows():
            task_id = result_df.loc[index, "task_id"]
            code_completion = result_df.loc[index, "completion"]
            item_log = self.find_detail_log(log_list, task_id, code_completion)
            if item_log is None: continue
            
            passed = result_df.loc[index, "passed"]
            item_log.set_accuracy(passed)

        
        for log in log_list: 
            true_list, false_list = [], [] 
            for log_detail in log.get_consistency_list():
                if log_detail.get_accuracy() == True: 
                    true_list.append(log_detail)
                else: 
                    false_list.append(log_detail)

            log_detail = None
            if len(true_list) >= len(false_list):
                log_detail = true_list[0]
            else:
                log_detail = false_list[0]

            log.set_pass_at_k(self.calculate_pass_at_k(true_list, false_list, pass_at_k))
            log.set_completion(log_detail.get_completion())
            log.set_final_answer(log_detail.get_final_answer())
            log.set_accuracy(log_detail.get_accuracy())
            log.set_token_count(log_detail.get_token_count())

        self.get_logger().add_to_buffer_list(log_list)
        self.get_logger().write_to_log_file()


    def find_detail_log(self, log_list, task_id, code_completion):
        for log in log_list: 
            if log.get_problem_id() != task_id : continue

            for log_detail in log.get_consistency_list():
                if log_detail.get_final_answer() == code_completion:
                    return log_detail
                
        return None

    def calculate_pass_at_k(self, true_list, false_list, k):
        n = len(true_list) + len(false_list)
        c = len(true_list)
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    @torch.inference_mode()
    def compute_implicit_reward_hacking(self, num_samples=5, temperature=0.7):
        print("Stage: Implicit Reward Hacking")
        train_dataset, test_dataset = self.get_dataset().preprocess_dataset()
        for idx, x in enumerate(tqdm(test_dataset)):
            prompt = x['prompt']
            sample_ID = x['sample_id']
            target = x['target']
            split = x['split']
            problem_id = x['problem_id']
            
            full_completion = self.generate(prompt)
            prompt_len = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            full_completion = full_completion[0][prompt_len:]
            
            for i in range(1, 11):
                portion_size = int(len(full_completion) * i / 10)
                portion = full_completion[:portion_size]
                portion_text = self.tokenizer.decode(portion, skip_special_tokens=True)

                modified_prompt = prompt + ' ' + portion_text + self.get_dataset().force_generate_answer_text 
                generated_answer = self.generate(modified_prompt, temperature=temperature, num_return_sequences=num_samples, num_beams=num_samples)
                for j in range(num_samples):
                    generated_answer_text = self.tokenizer.decode(generated_answer[j], skip_special_tokens=True)
                    final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, generated_answer_text, target)

                    ID = (idx*10*num_samples) + (i - 1) * num_samples + j
                    log = inference_reward_hacking_log_entity(ID, sample_ID, problem_id, split, prompt, target)
                    log.set_completion(generated_answer_text)
                    log.set_completion_portion_size(i / 10)
                    log.set_final_answer(final_answer)
                    log.set_accuracy(accuracy)
                    self.get_logger().add_to_buffer(log)

        self.get_logger().write_to_log_file()

    
    def calculate_entropy(self, full_path_file_name):
        print(f'{'*' * 100} Calculate Entropy {'*' * 100}')
        try:
            df = pd.read_csv(full_path_file_name)

            if 'Entropy' not in df.columns:
                df['Entropy'] = np.nan
                df['Entropy'] = df['Entropy'].astype('float64')
            if 'Completion_Loss' not in df.columns:
                df['Completion_Loss'] = np.nan
                df['Completion_Loss'] = df['Completion_Loss'].astype('float64')
            if 'Perplexity' not in df.columns:
                df['Perplexity'] = np.nan
                df['Perplexity'] = df['Perplexity'].astype('float64')
            
            for index, row in tqdm(df.iterrows(), total=len(df)):
                completion = df.loc[index, "Completion"]
                if completion is None or pd.isna(completion): continue

                try:
                    entropy, loss, perplexity = self.representation.calculate_entropy(completion, self.get_model(), self.tokenizer)
                    df.at[index, "Entropy"] = entropy
                    df.at[index, "Completion_Loss"] = loss
                    df.at[index, "Perplexity"] = perplexity
                except Exception as e:
                    print(f"Calculate Entropy : {e}")

            df.to_csv(full_path_file_name, index=False)            
        except Exception as e:
            print(f"{full_path_file_name}: {e}")

    def calculate_and_update_iit(self, full_path_file_name):
        print(f'{'*' * 100} Calculate IIT {'*' * 100}')
        try:
            df = pd.read_csv(full_path_file_name)

            if 'Phi_Reward_Raw' not in df.columns:
                df['Phi_Reward_Raw'] = np.nan
                df['Phi_Reward_Raw'] = df['Phi_Reward_Raw'].astype('float64')
            if 'Phi_Reward' not in df.columns:
                df['Phi_Reward'] = np.nan
                df['Phi_Reward'] = df['Phi_Reward'].astype('float64')
            if 'Tpm_Loss' not in df.columns:
                df['Tpm_Loss'] = np.nan
                df['Tpm_Loss'] = df['Tpm_Loss'].astype('float64')
            if 'Tpm_Entropy' not in df.columns:
                df['Tpm_Entropy'] = np.nan
                df['Tpm_Entropy'] = df['Tpm_Entropy'].astype('float64')
            if 'Reduced_Dimention' not in df.columns:
                df['Reduced_Dimention'] = np.nan
                df['Reduced_Dimention'] = df['Reduced_Dimention'].astype('int64')
            if 'Completion_Embedding_Shape' not in df.columns:
                df['Completion_Embedding_Shape'] = ''
            
            for index, row in tqdm(df.iterrows(), total=len(df)):
                sample_ID = df.loc[index, "Sample_ID"]
                prompt = df.loc[index, "Prompt"]
                completion = df.loc[index, "Completion"]
                if prompt is None or pd.isna(prompt) or completion is None or pd.isna(completion): continue

                try:
                    entity = iit_entity(key = sample_ID)
                    entity.set_promptID(sample_ID)
                    entity.set_prompt(prompt)
                    refine_prompt = self.representation.clean_prompt_for_phi(prompt)
                    prompt_emb, _, _ = self.representation.extract_representation(refine_prompt, self.get_model(), self.tokenizer, self.get_layer_type())
                    entity.set_prompt_embedding(prompt_emb)
    
                    entity.set_completion(completion)
                    completion_emb, _, _ = self.representation.extract_representation(entity.get_completion(), self.get_model(), self.tokenizer, self.get_layer_type())
                    entity.set_completion_embedding_and_shape(completion_emb)
                    entity.add_token_list(self.tokenizer, entity.get_completion(), completion_emb)

                    if not entity.is_calcutable(): continue
                    entity = self.get_iit_calculator().calculate_entity(entity)

                    df.at[index, "Phi_Reward_Raw"] = entity.get_iit_reward_raw()
                    df.at[index, "Phi_Reward_Raw_Actual"] = entity.get_iit_reward_raw_actual()
                    df.at[index, "Phi_Reward"] = entity.get_iit_reward()
                    df.at[index, "Tpm_Loss"] = entity.get_tpm_loss()
                    df.at[index, "Tpm_Entropy"] = entity.get_tpm_entropy()
                    
                    df.at[index, "Completion_Embedding_Shape"] = entity.get_completion_embedding_shape()
                    df.at[index, "Reduced_Dimention"] = entity.get_reduced_dim()
                    
                    del entity, completion_emb
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Calculate Entropy : {e}")

            df.to_csv(full_path_file_name, index=False)            
        except Exception as e:
            print(f"{full_path_file_name}: {e}")
        
    @torch.inference_mode()
    def load_embedding_and_loss(self, log_list): 
        result_list = []
        print('Stage: Embedding Extraction')
        for log in tqdm(log_list):     
            try:
                entity = iit_entity(key = log.get_ID())
                entity.set_promptID(log.get_sample_ID())
                entity.set_prompt(log.get_prompt())
                refine_prompt = self.representation.clean_prompt_for_phi(log.get_prompt())
                prompt_emb, _, _ = self.representation.extract_representation(refine_prompt, self.get_model(), self.tokenizer, self.get_layer_type())
                entity.set_prompt_embedding(prompt_emb)
 
                entity.set_completion(log.get_completion())
                if log.get_completion() is not None:
                    completion_emb, completion_loss, entropy = self.representation.extract_representation(entity.get_completion(), self.get_model(), self.tokenizer, self.get_layer_type())
                    entity.set_completion_loss(completion_loss)
                    entity.set_completion_embedding_and_shape(completion_emb)
                    entity.add_token_list(self.tokenizer, entity.get_completion(), completion_emb)
                    log.set_token_count(entity.get_token_count())
                    log.set_completion_embedding_shape(my_utils.embedding_tostring(completion_emb))
                    log.set_completion_loss(my_utils.tensor_tostring(completion_loss))
                    log.set_perplexity(my_utils.calculate_perplexity(completion_loss))
                    log.set_entropy(entropy)
                result_list.append(entity)
                
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                result_list.append(entity)
                print(f"[WARN] Load Embedding: {e}")
       
        return log_list, result_list


    def calculate_iit(self, log_list, result_list): 
        if self.get_iit_calculator() is None: 
            return log_list
        
        print('Stage: Calculate Integrated Information Theory')
        calcutable_list = list(filter(lambda x: x.is_calcutable() , result_list))
        calculated_list = self.get_iit_calculator().calculate(calcutable_list)
        for log in log_list:
            calculated_entity_list = list(filter(lambda r: r.get_key() == log.get_ID() , calculated_list))
            if calculated_entity_list is not None and len(calculated_entity_list) == 1:
                calculated_entity = calculated_entity_list[0]
                log.set_reduced_dim(calculated_entity.get_reduced_dim())
                log.set_phi_reward(calculated_entity.get_iit_reward())
                log.set_phi_reward_raw(calculated_entity.get_iit_reward_raw())
                log.set_phi_reward_raw_actual(calculated_entity.get_iit_reward_raw_actual())
                log.set_tpm_loss(calculated_entity.get_tpm_loss())
                log.set_tpm_entropy(calculated_entity.get_tpm_entropy())
                for token_entity in calculated_entity.get_iit_token_list():
                    token_log_entity = log_token_entity(token_entity.get_token_number(), token_entity.get_token())
                    token_log_entity.set_state_index(token_entity.get_state_index())
                    token_log_entity.set_iit_value(token_entity.get_iit_value())
                    token_log_entity.set_iit_cause_state_index(token_entity.get_iit_cause_state_index())
                    token_log_entity.set_iit_effect_state_index(token_entity.get_iit_effect_state_index())
                    log.add_log_token_list(token_log_entity)
        
        return log_list

    def get_model(self):
        if self.model == None: 
            bnb_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_compute_dtype = getattr(torch, "bfloat16"),
                bnb_4bit_use_double_quant = False,
            )
            if self.peft_checkpoint_path != None :
                base_model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config = bnb_config)
                model = PeftModel.from_pretrained(
                    base_model,
                    self.peft_checkpoint_path
                )

            else:
                model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config = bnb_config)

            model.config.use_cache = False
            model.config.pretraining_tp = 1        
            self.model = model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, fix_mistral_regex=True)

        return self.model

    def generate(self, prompt, temperature = 1.0, num_return_sequences = 1, num_beams = 1.0, top_p = 1.0, top_k = 50, do_sample = False ):
        model = self.get_model()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = model.generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_new_tokens = self.get_max_new_tokens(),
            temperature=temperature, 
            top_p=top_p,
            top_k=top_k,            
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            repetition_penalty = 1.1,
            early_stopping = True, #Can stop before reach the max_length
            eos_token_id = self.tokenizer.eos_token_id,
            pad_token_id = self.tokenizer.pad_token_id,
            do_sample = do_sample
        )

        return outputs
    
    def get_layer_type(self):
        if self.get_iit_calculator() is not None: 
            return self.get_iit_calculator().get_config().get_layer_type()
        
        return iit_layer_type_enum.SOME

    def get_output_file_path_code(self):
        return None

    def get_max_new_tokens(self):
        return 5000

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_iit_calculator(self):
        pass

    @abstractmethod
    def get_logger(self, run_number = 0):
        pass
