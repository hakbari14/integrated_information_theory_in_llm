from trl import GRPOConfig, ModelConfig
from integrated_information_theory.training.peft.open_thoughts_qwen3_8b.grpo_open_thoughts_qwen3_8_peft_trainer import grpo_open_thoughts_qwen3_8_peft_trainer
from integrated_information_theory.integrated_information import integrated_information
from integrated_information_theory.config.integrated_information_config import integrated_information_config
from integrated_information_theory.logger.training.training_logger import training_logger
from integrated_information_theory.enums_class import ii_phi_type_enum, iit_log_type_enum, granularity_enum, last_layer_computation_type_enum, tpm_creation_type_enum, ii_calculation_type_enum, training_type_enum, iit_layer_type_enum, iit_threashold_type_enum

class grpo_open_thoughts_qwen3_8_settings_85(grpo_open_thoughts_qwen3_8_peft_trainer): 

    def __init__(self, training_type):
        super().__init__(training_type)

    def get_model_config(self):

        if self.model_config is None:
            self.model_config = ModelConfig(
                model_name_or_path = self.model_name,
                attn_implementation="flash_attention_2",
                use_peft=True,
                lora_r=2048,
                lora_alpha=1024,
                load_in_4bit=True,
            )

        return self.model_config

    def get_training_args(self):
        
        if self.training_args is None:
            self.training_args = GRPOConfig(
                output_dir="live_logs/settings_85",
                learning_rate=3e-6,
                lr_scheduler_type="cosine",
                logging_steps=10,
                max_steps=1200,
                per_device_train_batch_size=2,      
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=1,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                bf16=True,
                use_vllm=True,
                vllm_mode="server",
                vllm_server_host="localhost",
                vllm_server_port=8000,
                max_completion_length=5000, 
                num_generations=2,                      
                num_generations_eval=1,                      
                beta=0.001,
                warmup_ratio=0.0,

                report_to=['tensorboard'],
                logging_dir='live_logs/settings_85/tb_logs',  
                eval_strategy="steps",  
                eval_steps=50,
                save_steps=50,
            )

        return self.training_args
    
    def get_iit_calculator(self):
        
        if self.iit_calculator is None:
            config = integrated_information_config()
            config.set_phi_type(ii_phi_type_enum.BIG_PHI)
            config.set_adaptive_dim(False)
            config.set_reduced_dim(4)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            config.set_granularity(granularity_enum.TOKEN)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)

            self.iit_calculator = integrated_information(config)

        return self.iit_calculator

    def get_logger(self):
        if self.logger is None:
            self.logger = training_logger(
                log_file_name = 'live_logs/settings_85/settings_85.csv',  
                log_type = iit_log_type_enum.TRAIN_TEST
            )

        return self.logger


t = grpo_open_thoughts_qwen3_8_settings_85(training_type_enum.IIT)
t.train()
