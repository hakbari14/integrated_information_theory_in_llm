from integrated_information_theory.training.grpo_trainer import grpo_trainer

class grpo_peft_trainer(grpo_trainer): 

    def __init__(self, model_name, training_type):
        super().__init__(model_name, training_type)

    def get_trainer(self):
        trainer = super().get_trainer()    
        trainer.model.print_trainable_parameters() 
        return trainer


