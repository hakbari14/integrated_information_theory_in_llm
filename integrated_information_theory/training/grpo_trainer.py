from trl import GRPOTrainer, get_peft_config
from abc import ABC, abstractmethod
from integrated_information_theory.logger.training.training_logger import training_logger
from integrated_information_theory.logger.training.training_log_entity import training_log_entity
from integrated_information_theory.entity.iit_entity import iit_entity
from integrated_information_theory.enums_class import training_type_enum, llm_pipeline_type_enum, iit_layer_type_enum
from integrated_information_theory.llm_representation import llm_representation
from integrated_information_theory.utils import my_utils
import re
import torch
import gc
import math 
import numpy as np

class grpo_trainer(ABC): 

    def __init__(self, model_name, training_type):
        self.model_name = model_name
        self.pipeline_type = llm_pipeline_type_enum.TRAINING
        if self.model_name is None:
            raise Exception('model name is required')

        self.training_type = training_type
        if self.training_type is None:
            raise Exception('training type is required')
        if training_type_enum.BASELINE != self.training_type and training_type_enum.IIT != self.training_type and training_type_enum.ENTROPY != self.training_type:
            raise Exception('training type has not been correctly determined')

        self.representation = llm_representation()
        self.dataset = None
        config = self.get_dataset().get_config()
        config.set_pipeline_type(llm_pipeline_type_enum.TRAINING)
        self.model_config = None
        self.training_args = None
        self.iit_calculator = None
        self.trainer = None
        self.logger = None
        self.old_step_accuracy = None
        self.current_step_accuracy = None
        self.old_step_tokens_count = None
        self.current_step_tokens_count = None

    def train(self):
        trainer = self.get_trainer()
        trainer.model.config.use_cache = False
        trainer.train()

    def resume_train(self):
        trainer = self.get_trainer()
        trainer.model.config.use_cache = False
        trainer.train(resume_from_checkpoint=True)

    def get_trainer(self):
        if self.trainer is None:
            train_dataset, eval_dataset = self.get_dataset().preprocess_dataset()
            model_config = self.get_model_config()
            
            self.trainer = GRPOTrainer(
                model = self.model_name,
                reward_funcs = self.get_reward_funcs(),
                args = self.get_training_args(),
                train_dataset = train_dataset,
                eval_dataset = eval_dataset,
                peft_config = get_peft_config(model_config),
            )
            
        return self.trainer

    def get_reward_funcs(self):
        if training_type_enum.IIT == self.training_type:
            return [self.accuracy_reward, self.calculate_iit_reward]
        elif training_type_enum.BASELINE == self.training_type:
            return [self.accuracy_reward]
        elif training_type_enum.ENTROPY == self.training_type:
            return [self.accuracy_reward, self.calculate_entropy_reward]
        else:
            return []

    @torch.inference_mode()
    def calculate_iit_reward(self, completions, target=None, tokenizer=None, **kwargs):
        """
        Reward function that uses the *currently training model* to compute Phi-based reward.
        Also:
        - Logs (step, prompt, completion, phi) to CSV_PATH (all phases)
        - Sends eval samples to EVAL_CSV_PATH together with accuracy_reward via buffer
        """

        prompts = kwargs.get("prompts") or kwargs.get("prompt") or kwargs.get("inputs")
        split_list = kwargs.get("split")     # list[str], e.g. "train"/"eval"
        sample_ids = kwargs.get("sample_id") # list[int]
        problem_ids = kwargs.get("problem_id", None)
        trainer_state = kwargs.get("trainer_state", None)

        trainer = self.get_trainer()
        model = trainer.model
        tokenizer = trainer.processing_class

        rewards = []
        result_list = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            prompt = prompts[i]
            sample_ID = sample_ids[i]
            entity = iit_entity(key=i)
            entity.set_promptID(sample_ID)
            entity.set_prompt(prompt)
            entity.set_completion(completion)

            if len(completion.split()) == 0:
                result_list.append(entity)
                continue

            try:
                refine_prompt = self.representation.clean_prompt_for_phi(prompt)
                prompt_emb, prompt_loss = self.representation.extract_representation(refine_prompt, model, tokenizer, self.get_layer_type())
                entity.set_prompt_embedding(prompt_emb)
                completion_emb, completion_loss = self.representation.extract_representation(completion, model, tokenizer, self.get_layer_type())
                entity.set_completion_loss(completion_loss)
                entity.set_completion_embedding_and_shape(completion_emb)
                entity.add_token_list(tokenizer, completion, completion_emb)
                result_list.append(entity)

                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                result_list.append(entity)
                print(f"[WARN] calculate_iit_reward[create iit_entity]: {e}")
        
        self.calculate_coefficient_mean_std(result_list)        
        calcutable_list = list(filter(lambda x: x.is_calcutable() , result_list))
        calculated_list = self.get_iit_calculator().calculate(calcutable_list)
        for x in result_list:
            try:
                i = x.get_key()
                split = split_list[i]
                problem_id = problem_ids[i] if problem_ids is not None else None
                trainer_global_step = trainer_state.global_step
                sample_ID = x.get_promptID()
                prompt = x.get_prompt()
                ground_truth = target[i]
                completion = x.get_completion()
                log = training_log_entity(sample_ID, problem_id, split, trainer_global_step, prompt, ground_truth, completion)
                log.set_token_count(x.get_token_count())
                log.set_completion_embedding_shape(x.get_completion_embedding_shape())
                log.set_completion_loss(my_utils.tensor_tostring(x.get_completion_loss()))
                log.set_perplexity(my_utils.calculate_perplexity(x.get_completion_loss()))

                calculated_entity_list = list(filter(lambda r: r.get_key() == x.get_key() , calculated_list))
                if calculated_entity_list is not None and len(calculated_entity_list) == 1:
                    calculated_entity = calculated_entity_list[0]
                    log.set_token_count_for_reduced_dim(calculated_entity.get_token_count_for_reduced_dim())
                    log.set_reduced_dim(calculated_entity.get_reduced_dim())
                    log.set_phi_reward(calculated_entity.get_iit_reward())
                    log.set_phi_reward_raw(calculated_entity.get_iit_reward_raw())

                self.get_logger().add_to_buffer(log)
                rewards.append(log.get_phi_reward())
            except Exception as e:
                print(f"[Error] calculate_iit_reward[create log]: {e}")

        self.get_logger().write_to_log_file()
        return rewards

    def accuracy_reward(self, completions, target, **kwargs):
        rewards = []
        split_list = kwargs.get("split")     
        sample_ids = kwargs.get("sample_id") 
        problem_ids = kwargs.get("problem_id", None)
        prompts = kwargs.get("prompts") or kwargs.get("prompt") or kwargs.get("inputs")
        trainer_state = kwargs.get("trainer_state", None)

        trainer = self.get_trainer()
        model = trainer.model
        tokenizer = trainer.processing_class

        for i, (completion, gt) in enumerate(zip(completions, target)):
            split = split_list[i]
            sample_ID = sample_ids[i]
            problem_id = problem_ids[i] if problem_ids is not None else None
            prompt = prompts[i]
            trainer_global_step = trainer_state.global_step
            log = training_log_entity(sample_ID, problem_id, split, trainer_global_step, prompt, gt, completion)

            try:
                if training_type_enum.BASELINE == self.training_type:
                    completion_loss = self.representation.extract_loss(completion, model, tokenizer)
                    log.set_completion_loss(my_utils.tensor_tostring(completion_loss))
                    log.set_perplexity(my_utils.calculate_perplexity(completion_loss))

                answer, target_answer_equal, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(completion, gt)
                if answer is None:
                    acc_reward = 0.0
                else:
                    acc_reward = 1.0 if target_answer_equal else 0.0

                rewards.append(acc_reward)
                log.set_accuracy(acc_reward == 1.0)
                log.set_accuracy_reward(acc_reward)
                log.set_final_answer(answer)
            except Exception:
                log.set_accuracy(False)
                log.set_accuracy_reward(0.0)
                rewards.append(0.0)
            
            self.get_logger().add_to_buffer(log)

        self.old_step_accuracy = self.current_step_accuracy
        self.current_step_accuracy = np.mean(rewards)
        if training_type_enum.BASELINE == self.training_type:
            self.get_logger().write_to_log_file()

        return rewards

    @torch.inference_mode()
    def calculate_entropy_reward(self, completions, target=None, tokenizer=None, **kwargs):
        prompts = kwargs.get("prompts") or kwargs.get("prompt") or kwargs.get("inputs")
        split_list = kwargs.get("split")     # list[str], e.g. "train"/"eval"
        sample_ids = kwargs.get("sample_id") # list[int]
        problem_ids = kwargs.get("problem_id", None)
        trainer_state = kwargs.get("trainer_state", None)

        trainer = self.get_trainer()
        model = trainer.model
        tokenizer = trainer.processing_class

        rewards = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            split = split_list[i]
            sample_ID = sample_ids[i]
            problem_id = problem_ids[i] if problem_ids is not None else None
            prompt = prompts[i]
            trainer_global_step = trainer_state.global_step
            ground_truth = target[i]
            log = training_log_entity(sample_ID, problem_id, split, trainer_global_step, prompt, ground_truth, completion)

            try:
                completion_loss = self.representation.extract_loss(completion, model, tokenizer)
                log.set_completion_loss(my_utils.tensor_tostring(completion_loss))
                loss = completion_loss.item()
                reward = 1 if loss == 0 else 1 - math.exp(-1.0 / loss)
                log.set_entropy_reward(reward)
                rewards.append(reward)

                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                rewards.append(0.0)
                log.set_entropy_reward(0.0)
                print(f"[WARN] calculate_entropy_reward: {e}")

            self.get_logger().add_to_buffer(log)
    
        self.get_logger().write_to_log_file()
        return rewards


    def calculate_coefficient_mean_std_2(self, iit_entity_list):
        tokens_count = 0
        for entity in iit_entity_list:
            if entity.get_completion_embedding() is not None: 
                tokens_count += entity.get_completion_embedding().shape[1]
        
        self.old_step_tokens_count = self.current_step_tokens_count
        self.current_step_tokens_count = tokens_count
        W_mean, W_std = 0.5, 0.5
        if self.old_step_tokens_count is not None and self.old_step_accuracy is not None: 
           if self.old_step_tokens_count > self.current_step_tokens_count and self.old_step_accuracy > self.current_step_accuracy:
                W_mean, W_std = 0.3, 0.7
           if self.old_step_tokens_count < self.current_step_tokens_count and self.old_step_accuracy < self.current_step_accuracy:
                W_mean, W_std = 0.7, 0.3

        print()
        print(f'OLD Accuracy = {self.old_step_accuracy}, Cuurent Accuacy = {self.current_step_accuracy}')
        print(f'OLD Tokens Count = {self.old_step_tokens_count}, Cuurent Tokens Count = {self.current_step_tokens_count}')
        print()

        self.get_iit_calculator().get_config().set_coefficient_mean_reward_dimension(W_mean)
        self.get_iit_calculator().get_config().set_coefficient_std_reward_dimension(W_std)
        

    def calculate_coefficient_mean_std(self, iit_entity_list):
        if self.get_iit_calculator().get_config().get_adaptive_dim() is None or self.get_iit_calculator().get_config().get_adaptive_dim() == False: 
            self.get_iit_calculator().get_config().set_coefficient_mean_reward_dimension(0.5)
            self.get_iit_calculator().get_config().set_coefficient_std_reward_dimension(0.5)
            return None 
        if self.get_iit_calculator().get_config().get_is_fixed_coefficient() == True: 
            return None 
        
        tokens_count = 0
        for entity in iit_entity_list:
            if entity.get_completion_embedding() is not None: 
                tokens_count += entity.get_completion_embedding().shape[1]
        
        self.old_step_tokens_count = self.current_step_tokens_count
        self.current_step_tokens_count = tokens_count

        print()
        W_mean, W_std = 0.5, 0.5
        if self.old_step_tokens_count is not None and self.old_step_accuracy is not None: 
            W_token =  (self.old_step_tokens_count - self.current_step_tokens_count) / self.old_step_tokens_count
            W_accuracy =  (self.current_step_accuracy - self.old_step_accuracy) / self.old_step_accuracy

            if W_token * W_accuracy < 0: 
                T = abs(W_token) + abs(W_accuracy)
                Z = abs(W_token) / T if abs(W_token) > abs(W_accuracy) else abs(W_accuracy) / T
            else: 
                diff = abs(W_token - W_accuracy)
                _min = min(abs(W_token), abs(W_accuracy))
                _max = max(abs(W_token), abs(W_accuracy))
                if _max != 0:
                    Z = diff / _max if diff > _min else _min / _max
                else: 
                    Z = 0.5

            if W_token > W_accuracy:
                W_std = Z
                W_mean = 1.0 - Z
            elif W_token < W_accuracy:
                W_mean = Z
                W_std = 1.0 - Z 
            else:
                W_mean = 0.5
                W_std = 0.5

            print(f'W_Token = {W_token}, W_accuracy = {W_accuracy}')

        print(f'OLD Accuracy = {self.old_step_accuracy}, Cuurent Accuacy = {self.current_step_accuracy}')
        print(f'OLD Tokens Count = {self.old_step_tokens_count}, Cuurent Tokens Count = {self.current_step_tokens_count}')
        print()

        self.get_iit_calculator().get_config().set_coefficient_mean_reward_dimension(W_mean)
        self.get_iit_calculator().get_config().set_coefficient_std_reward_dimension(W_std)
        return None 
        

    def contains_foreign_language(self, text: str) -> bool:
        """
        Returns True if the text contains characters
        outside typical Persian, Arabic, English, numbers, and punctuation.

        Specifically detects CJK (Chinese, Japanese, Korean) or other scripts.
        """
        # Unicode ranges for CJK (Chinese, Japanese, Korean)
        cjk_pattern = re.compile(
            r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF"  # CJK Unified Ideographs
            r"\u3040-\u30FF\u31F0-\u31FF"                # Hiragana + Katakana
            r"\uAC00-\uD7AF]"                            # Hangul syllables
        )

        # English, Persian/Arabic letters, digits, whitespace, punctuation
        allowed_pattern = re.compile(
            r"^[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"  # Arabic script (Persian subset)
            r"a-zA-Z0-9\s.,!?;:'\"()\[\]{}<>@#$%^&*=_+\-/\\|`~،؛؟»«٪]*$"
        )

        if cjk_pattern.search(text):
            # clearly contains Chinese/Japanese/Korean
            return True

        # If text has any character not in allowed set
        if not allowed_pattern.match(text):
            return True

        return False

    def get_layer_type(self):
        if self.get_iit_calculator() is not None: 
            return self.get_iit_calculator().get_config().get_layer_type()
        
        return iit_layer_type_enum.SOME

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_model_config(self):
        pass

    @abstractmethod
    def get_training_args(self):
        pass

    @abstractmethod
    def get_iit_calculator(self):
        pass

    @abstractmethod
    def get_logger(self):
        pass

