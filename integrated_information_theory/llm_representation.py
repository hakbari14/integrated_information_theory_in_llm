from integrated_information_theory.utils import my_utils
from integrated_information_theory.enums_class import iit_layer_type_enum
import re
import gc
import torch
import numpy as np


class llm_representation: 

    def extract_representation(self, text: str, model, tokenizer, layer_type: iit_layer_type_enum) -> tuple[np.ndarray, float, float]:
        """Return hidden states from the given model (no gradients)."""
        # model.eval()  
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt').to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True)
            hidden = torch.cat(outputs.hidden_states, dim=0).detach().cpu().float().numpy()
            loss = outputs.loss.cpu()
            logits = outputs.logits.squeeze(0).detach().float()
            entropy = my_utils.calculate_entropy(logits)
            
            del outputs, inputs, logits
            torch.cuda.empty_cache()
            
            if iit_layer_type_enum.SOME == layer_type: 
                sampled_layers = np.linspace(1, hidden.shape[0]-1, num=12)
                sampled_layers = [int(x) for x in np.round(sampled_layers).tolist()]
                layer_2_3 = int(round((hidden.shape[0]-1) * 2 / 3))
                if layer_2_3 not in sampled_layers:
                    sampled_layers.append(layer_2_3)
                sampled_layers = sorted(set(sampled_layers))
                filtered_hidden = hidden[sampled_layers, :, :].copy()
                del hidden
                return filtered_hidden, loss.item(), entropy
            
            elif iit_layer_type_enum.ALL == layer_type: 
                return hidden, loss.item(), entropy

            elif iit_layer_type_enum.LAST == layer_type: 
                filtered_hidden = hidden[-1, :, :].copy()
                del hidden
                return filtered_hidden, loss.item(), entropy

        return None, None, None
    
    def optimize_extract_representation(self, text: str, model, tokenizer, layer_type: iit_layer_type_enum) -> tuple[np.ndarray, float, float]:
        """Return hidden states from the given model (no gradients)."""
        inputs = outputs = None
        try:
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors='pt').to(model.device)
                outputs = model(
                    **inputs, labels=inputs["input_ids"],
                    output_hidden_states=True, use_cache=False,
                )
                num_layers = len(outputs.hidden_states)
                if layer_type == iit_layer_type_enum.SOME:
                    sampled_layers = np.linspace(1, num_layers - 1, num=12)
                    sampled_layers = [int(x) for x in np.round(sampled_layers).tolist()]
                    layer_2_3 = int(round((num_layers - 1) * 2 / 3))
                    if layer_2_3 not in sampled_layers:
                        sampled_layers.append(layer_2_3)
                    layer_indices = sorted(set(sampled_layers))
                elif layer_type == iit_layer_type_enum.ALL:
                    layer_indices = range(num_layers)
                elif layer_type == iit_layer_type_enum.LAST:
                    layer_indices = [num_layers - 1]
                else:
                    return None, None, None

                # Transfer only requested layers; never concatenate them on GPU.
                hidden = np.concatenate([
                    outputs.hidden_states[i].detach().cpu().float().numpy()
                    for i in layer_indices
                ], axis=0)
                loss = outputs.loss.item()
                # Convert one token's logits at a time, avoiding a full FP32 copy.
                entropy = my_utils.calculate_entropy(
                    row.detach().float() for row in outputs.logits.squeeze(0)
                )
                if layer_type == iit_layer_type_enum.LAST:
                    hidden = hidden[0].copy()
                return hidden, loss, entropy
        finally:
            # Also release tensors when extraction raises (including CUDA OOM).
            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def extract_representation_last_layer(self, text, model, tokenizer):
        """Return hidden states from the given model (no gradients)."""
        # model.eval()  
        # with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt').to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True)
        hidden = torch.cat(outputs.hidden_states, dim=0).detach().cpu().float().numpy()
        loss = outputs.loss.cpu()
        
        del outputs, inputs
        torch.cuda.empty_cache()
        
        # sample layers (you can tune this for speed)
        return np.mean(hidden[hidden.shape[0]-1], axis=0), loss

    def extract_loss(self, text, model, tokenizer):
        """Return hidden states from the given model (no gradients)."""
        # model.eval()  
        # with torch.no_grad():

        inputs = tokenizer(text, return_tensors='pt').to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.cpu()

        del outputs, inputs
        torch.cuda.empty_cache()

        return loss

    def calculate_entropy(self, text, model, tokenizer):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt').to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits.squeeze(0).detach().float()
            entropy = my_utils.calculate_entropy(logits)
            loss = my_utils.tensor_tostring(outputs.loss.cpu())
            perplexity = my_utils.calculate_perplexity(outputs.loss.cpu())
            

        del outputs, inputs, logits
        torch.cuda.empty_cache()
        gc.collect()
        return entropy, loss, perplexity

    def compute_conditional_loss(self, model, tokenizer, context, target):
        # Combine context + target
        full_text = context + target

        # Tokenize
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

        # Create labels
        labels = input_ids.clone()

        # Mask out context tokens (we only compute loss on target)
        context_ids = tokenizer(context, return_tensors="pt")["input_ids"]
        context_length = context_ids.shape[1]

        labels[:, :context_length] = -100  # ignore context

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        return loss
    
    def clean_prompt_for_phi(self, raw_prompt: str) -> str:
        """
        1) Remove the <|im_start|>system ... <|im_end|> block.
        2) Extract the last 'User:' segment content.
        3) Drop boilerplate ('A conversation between ...').
        4) Remove any leftover <think>...</think> from the prompt (if present).
        5) Trim whitespace.
        """
        SYSTEM_BLOCK_RE = re.compile(
            r"<\|im_start\|>\s*system\b.*?<\|im_end\|>\s*", re.DOTALL | re.IGNORECASE
        )
        GENERIC_BLOCK_RE = re.compile(
            r"<\|im_start\|>.*?<\|im_end\|>\s*", re.DOTALL | re.IGNORECASE
        )

        p = SYSTEM_BLOCK_RE.sub("", raw_prompt)

        m = list(re.finditer(
            r"User:\s*(.*?)(?=\n<\|im_start\|>|\nAssistant:|$)",
            p,
            flags=re.DOTALL | re.IGNORECASE
        ))
        if m:
            text = m[-1].group(1)
        else:
            
            text = GENERIC_BLOCK_RE.sub("", p)

        # Remove boilerplate lines that sometimes appear before/around "User:"
        text = re.sub(
            r"^A conversation between.*?User:\s*",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE
        )

        # Remove any <think>...</think> that may be embedded in the prompt text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

        return text.strip()


