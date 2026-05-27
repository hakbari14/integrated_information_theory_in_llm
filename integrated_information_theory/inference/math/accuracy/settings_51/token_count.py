import pandas as pd
from transformers import AutoTokenizer
from tokenizers import Tokenizer as HFTokenizer
import os


def load_tokenizer():
   
    model_name = '/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60'
    return AutoTokenizer.from_pretrained(model_name)
    


def count_tokens(tokenizer, text):
    """Count tokens depending on tokenizer type."""
    if hasattr(tokenizer, "encode") and "transformers" in str(type(tokenizer)):
        # HuggingFace tokenizer
        return len(tokenizer.encode(text, add_special_tokens=False))
    else:
        # tokenizers.Tokenizer
        return len(tokenizer.encode(text).ids)


def add_token_counts(csv_path, output_path=None):
    # Load tokenizer
    tokenizer = load_tokenizer()

    # Load CSV
    df = pd.read_csv(csv_path)

    if "Completion" not in df.columns:
        raise ValueError("CSV must contain a 'Completion' column")

    # Handle NaNs safely
    df["Completion"] = df["Completion"].fillna("").astype(str)

    # Compute token counts
    df["Token_Count_Reduced_Dimention"] = df["Completion"].apply(
        lambda x: count_tokens(tokenizer, x)
    )

    # Output path
    if output_path is None:
        base, ext = os.path.splitext(csv_path)
        output_path = f"{base}_with_token_counts{ext}"

    # Save CSV
    df.to_csv(output_path, index=False)

    print(f"Saved updated CSV to: {output_path}")


# Example usage:
# add_token_counts("./my_tokenizer", "./data.csv")
add_token_counts("/home/hr_akbari/research/LLM_PostTraining/integrated_information_theory/inference/math/accuracy/settings_0/settings_0_math500_full.csv", "/home/hr_akbari/research/LLM_PostTraining/integrated_information_theory/inference/math/accuracy/settings_0/settings_0_math500_full.csv")