import torch 
import math
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer

# ref: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llmtuner/model/patcher.py
def noisy_mean_initialization(embed_weight: torch.Tensor, num_new_tokens: int, generator):
    embedding_dim = embed_weight.size(1)  # (V, H) 
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)  # (1, H)
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])  # (num_new_tokens, H)
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)), generator=generator)
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight

# TODO: other ways to initialize new tokens?

def resize_embedding_layer(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, generator=None) -> None:
    """
    Resize token embeddings.
    """
    current_embedding_size = model.get_input_embeddings().weight.size(0)  # model.get_input_embeddings(): Embedding(V, H)

    if len(tokenizer) > current_embedding_size:
        if not isinstance(model.get_output_embeddings(), torch.nn.Linear):  # model.get_output_embeddings(): Linear(H, V)
            print("Current model does not support resizing token embeddings.")
            return

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

        new_embedding_size = model.get_input_embeddings().weight.size(0)
        num_new_tokens = new_embedding_size - current_embedding_size  # input_embeddings和output_embeddings的weight都是(V, H)
        noisy_mean_initialization(model.get_input_embeddings().weight.data, num_new_tokens, generator)
        noisy_mean_initialization(model.get_output_embeddings().weight.data, num_new_tokens, generator)

        print("Resized token embeddings from {} to {}.".format(current_embedding_size, new_embedding_size))

if __name__ == "__main__":
    model_input_dir = "base_models/tinyllama_3T"
    tokenizer_dir = "base_models/colossal_llama_2_7b"
    model_output_dir = "experiment_models/tinyllama_empty_expand"
    seed = 24

    g = torch.Generator()
    g.manual_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(model_input_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    resize_embedding_layer(model, tokenizer, generator=g)
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

