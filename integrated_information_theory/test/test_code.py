import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from integrated_information_theory.utils import my_utils

my_utils.convert_model_into_hugginfacces('/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60/', './live_logs/settings_64/checkpoint-500')
my_utils.convert_model_into_hugginfacces('/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60/','./live_logs/settings_65/checkpoint-1200')