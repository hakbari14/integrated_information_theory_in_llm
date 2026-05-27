import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging,)
from integrated_information_theory.entity.iit_entity import iit_entity
from integrated_information_theory.llm_representation import llm_representation
from integrated_information_theory.intrinsic_information import intrinsic_information
from integrated_information_theory.integrated_information import integrated_information
from integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from integrated_information_theory.config.integrated_information_config import integrated_information_config
from integrated_information_theory.enums_class import ii_phi_type_enum, ii_calculation_type_enum, tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum, iit_layer_type_enum, iit_threashold_type_enum
from peft import PeftModel
import torch
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


class iit_reward_heatmap(object):
    
    def __init__(self, problem_no, settings, calculator, model_name, peft_checkpoint_path=None):
        self.model_name = model_name
        self.peft_checkpoint_path = peft_checkpoint_path
        self.problem_no = problem_no
        self.settings = settings
        self.calculator = calculator

        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = getattr(torch, "bfloat16"),
            bnb_4bit_use_double_quant = False,
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config = bnb_config)
        if peft_checkpoint_path is not None: 
            self.model = PeftModel.from_pretrained(self.base_model, self.peft_checkpoint_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.peft_checkpoint_path)
        else:
            self.model = self.base_model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.representation = llm_representation()
        self.filename = 'integrated_information_theory/inference/analysis/iit_reward_heatmap/iit_reward_heatmap_data.csv'.replace('.csv', f'_{self.problem_no}.csv')

    def plot_heatmap(self, color_min, color_max, cmap="Reds"):
        entity_list = self.calculate_heatmap()
        if entity_list is None: 
            return None 
        
        for entity in entity_list:
            completion = []
            completion_score = []
            for idx, token in enumerate(entity.get_iit_token_list()):
                completion.append(token.get_token())
                completion_score.append(token.get_iit_value())

            self.plot_word_heatmap(completion, completion_score, color_min, color_max)
            
        return None

    def calculate_heatmap(self):
        df = pd.read_csv(self.filename)
        for index, row in df.iterrows():
            settings = df.loc[index, "Settings"]
            prompt = df.loc[index, "Prompt"]
            prompt_ID = df.loc[index, "Sample_ID"]
            completion = df.loc[index, "Completion"]
            
            if settings != self.settings: continue
            if self.get_iit_calculator() == None: continue

            layer_type = self.get_iit_calculator().get_config().get_layer_type()
            entity = iit_entity(key=index)
            entity.set_promptID(prompt_ID)
            entity.set_prompt(prompt)
            entity.set_completion(completion)
            refine_prompt = self.representation.clean_prompt_for_phi(entity.get_prompt())
            prompt_emb, prompt_loss = self.representation.extract_representation(refine_prompt, self.model, self.tokenizer, layer_type)
            entity.set_prompt_embedding(prompt_emb)

            refine_completion = self.representation.clean_prompt_for_phi(entity.get_completion())
            completion_emb, completion_loss = self.representation.extract_representation(entity.get_completion(), self.model, self.tokenizer, layer_type)
            entity.set_completion_loss(completion_loss)
            entity.set_completion_embedding_and_shape(completion_emb)
            entity.add_token_list(self.tokenizer, entity.get_completion(), completion_emb)

            iit_entity_list = []
            iit_entity_list.append(entity)
            iit_entity_list = self.get_iit_calculator().calculate(iit_entity_list)
            return iit_entity_list

        return None

    def plot_word_heatmap(self, words, scores, color_min, color_max, cmap="Reds"):
        
        # norm = mpl.colors.Normalize(vmin=min(scores), vmax=max(scores))
        norm = mpl.colors.Normalize(vmin=color_min, vmax=color_max)
        cmap = plt.get_cmap(cmap)

        # create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")

        # start position in axis coordinates
        x0, y0 = 0.01, 0.98
        x, y = x0, y0

        # spacing config
        line_spacing = 1.4  # multiplier for font size spacing
        fontsize = 12

        # need renderer for text size measurement
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        ax_bbox = ax.get_window_extent(renderer=renderer)
        ax_width_px = ax_bbox.width

        for word, score in zip(words, scores):
            color = cmap(norm(score))

            # create a temporary text object to measure width
            text = ax.text(
                0, 0,
                word + " ",
                fontsize=fontsize,
                transform=ax.transAxes
            )

            fig.canvas.draw()
            bbox = text.get_window_extent(renderer=renderer)
            word_width = bbox.width

            text.remove()  # remove temporary measurement text

            # convert pixel width to axis fraction
            word_width_ax = word_width / ax_width_px

            # wrap line if needed
            if x + word_width_ax > 0.99:
                x = x0
                y -= (fontsize / 72) / fig.get_size_inches()[1] * line_spacing

            # draw final word
            ax.text(
                x, y,
                word + " ",
                fontsize=fontsize,
                color=color,
                ha="left",
                va="top",
                transform=ax.transAxes
            )

            x += word_width_ax

        plt.tight_layout()
        plt.plot()
        plt.savefig(f'./integrated_information_theory/inference/analysis/iit_reward_heatmap/{self.calculator.lower()}/{self.settings}_{self.problem_no}_heatmap.png')
        print(f'IMAGE SAVED {self.settings}, Min Score ={min(scores)}, Max Score = {max(scores)}')


    def get_iit_calculator(self):
        if 'Settings_46' == self.calculator:
            config = intrinsic_information_config()
            config.set_calculation_type(ii_calculation_type_enum.SUM)
            config.set_adaptive_dim(False)
            config.set_reduced_dim(5)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            config.set_granularity(granularity_enum.TOKEN)
            return intrinsic_information(config)
        
        elif 'Settings_64' == self.calculator:
            config = integrated_information_config()
            config.set_phi_type(ii_phi_type_enum.SYSTEM_PHI)
            config.set_adaptive_dim(False)
            config.set_reduced_dim(4)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(1.0)
            config.set_granularity(granularity_enum.TOKEN)
            return integrated_information(config)
        
        elif 'Settings_65' == self.calculator:
            config = integrated_information_config()
            config.set_phi_type(ii_phi_type_enum.BIG_PHI)
            config.set_adaptive_dim(False)
            config.set_reduced_dim(4)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            config.set_granularity(granularity_enum.TOKEN)
            return integrated_information(config)

        return None

settings = '46'
problem_no = '29'
if '46' == settings:
    color_min, color_max = 0.6, 7.0
    h_46 = iit_reward_heatmap(problem_no, 'Settings_46', 'Settings_46', '/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', '/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_46/checkpoint-500')
    h_46.plot_heatmap(color_min, color_max)

    # h_0 = iit_reward_heatmap(problem_no, 'Settings_0', 'Settings_46', '/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', )
    # h_0.plot_heatmap(color_min, color_max)

    # h_37 = iit_reward_heatmap(problem_no, 'Settings_37', 'Settings_46', '/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', '/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_37/checkpoint-1200')
    # h_37.plot_heatmap(color_min, color_max)

    # h_51 = iit_reward_heatmap(problem_no, 'Settings_51', 'Settings_46', '/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', '/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_51/checkpoint-750')
    # h_51.plot_heatmap(color_min, color_max)

elif '64' == settings:
    color_min, color_max = 0.0, 0.4
    h_64 = iit_reward_heatmap(problem_no, 'Settings_64', 'Settings_64', '/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', '/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_64/checkpoint-500')
    h_64.plot_heatmap(color_min, color_max)

    h_0 = iit_reward_heatmap(problem_no, 'Settings_0', 'Settings_64', '/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', )
    h_0.plot_heatmap(color_min, color_max)

    h_37 = iit_reward_heatmap(problem_no, 'Settings_37', 'Settings_64', '/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', '/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_37/checkpoint-1200')
    h_37.plot_heatmap(color_min, color_max)

    h_51 = iit_reward_heatmap(problem_no, 'Settings_51', 'Settings_64', '/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', '/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_51/checkpoint-750')
    h_51.plot_heatmap(color_min, color_max)

elif '65' == settings:
    color_min, color_max = 0.0, 15.5
    h_65 = iit_reward_heatmap(problem_no, 'Settings_65', 'Settings_65', '/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', '/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_65/checkpoint-1200')
    h_65.plot_heatmap(color_min, color_max)

    h_0 = iit_reward_heatmap(problem_no, 'Settings_0', 'Settings_65', '/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', )
    h_0.plot_heatmap(color_min, color_max)

    h_37 = iit_reward_heatmap(problem_no, 'Settings_37', 'Settings_65', '/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', '/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_37/checkpoint-1200')
    h_37.plot_heatmap(color_min, color_max)

    h_51 = iit_reward_heatmap(problem_no, 'Settings_51', 'Settings_65', '/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60', '/home/hr_akbari/research/LLM_PostTraining/live_logs/settings_51/checkpoint-750')
    h_51.plot_heatmap(color_min, color_max)
