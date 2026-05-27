from integrated_information_theory.training.grpo_trainer import grpo_trainer
from integrated_information_theory.logger.training.training_log_entity import training_log_entity
from integrated_information_theory.enums_class import training_type_enum
from integrated_information_theory.utils import my_utils

class grpo_peft_trainer_countdown(grpo_trainer): 

    def __init__(self, model_name, training_type):
        super().__init__(model_name, training_type)

    def get_trainer(self):
        trainer = super().get_trainer()    
        trainer.model.print_trainable_parameters() 
        return trainer

    def accuracy_reward(self, completions, target, nums, **kwargs):
        rewards = []
        split_list = kwargs.get("split")     
        sample_ids = kwargs.get("sample_id") 
        problem_ids = kwargs.get("problem_id", None)
        prompts = kwargs.get("prompts") or kwargs.get("prompt") or kwargs.get("inputs")
        trainer_state = kwargs.get("trainer_state", None)

        trainer = self.get_trainer()
        model = trainer.model
        tokenizer = trainer.processing_class

        for i, (completion, gt, numbers) in enumerate(zip(completions, target, nums)):
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

                answer, target_answer_equal = self.get_dataset().extract_and_verify_final_answer(completion, gt, numbers)
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

        if training_type_enum.BASELINE == self.training_type:
            self.get_logger().write_to_log_file()

        return rewards


