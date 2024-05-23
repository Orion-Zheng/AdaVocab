from transformers import AutoModelForCausalLM
from adavocab_llama.ada_vocab_llama import AdaVocabLlamaForCausalLM
import torch

# Two model must be compared at the correct precision. 
# For example, original model is in float32 and has been trained in bfloat16. 
# If we compare original model and its checkpoint at float32/16 will lead to incorrect results.
# model_1 = AutoModelForCausalLM.from_pretrained("experiment_ckpts/tinyllama_expanded_frez_embed-2024-04-10-210707/checkpoint-10", 
#                                                torch_dtype=torch.bfloat16)
model_1 = AdaVocabLlamaForCausalLM.from_pretrained("original_models/ada-tinyllama-empty",
                                               torch_dtype=torch.bfloat16)
model_2 = AdaVocabLlamaForCausalLM.from_pretrained("experiment_ckpts/AdaVocab_debug-2024-05-23-102710/final_ckpt_backup-3",
                                               torch_dtype=torch.bfloat16)

comparison_dict = {}
def compare_models(model_1, model_2):
    for (name_1, param_1), (name_2, param_2) in zip(model_1.named_parameters(), model_2.named_parameters()):
        assert name_1 == name_2, f'Unmatched Parameter Names: {name_1}, {name_2}'
        if torch.allclose(param_1, param_2):   # equal?
            comparison_dict[name_1] = True
        else:
            print(f"Different Parameter Values at: {name_1}")
            comparison_dict[name_1] = False
            

print(model_1)
compare_models(model_1, model_2)
are_same = all(comparison_dict.values())
# print(comparison_dict)
if are_same:
    print("Two models are the same.")
else:
    print("Two models are different.")
