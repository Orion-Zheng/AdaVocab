from transformers import AutoTokenizer, AutoModelForCausalLM
from codebase.utils import find_all_linear_names
model_dir = "base_models/llama2-7b"
model = AutoModelForCausalLM.from_pretrained(model_dir)
print(find_all_linear_names(model, bits=32))  # ['o_proj', 'v_proj', 'k_proj', 'down_proj', 'lm_head', 'gate_proj', 'q_proj', 'up_proj']
