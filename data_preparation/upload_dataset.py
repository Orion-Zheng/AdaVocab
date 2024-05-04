from datasets import load_from_disk

# dataset = load_from_disk("original_datasets/skypile_2022_sampled_5000M")
# # dataset = dataset.map(...)  # do all your processing here
# dataset.push_to_hub("OrionZheng/skypile_2022_sampled_5000M",
#                     token="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",
#                     private=True)

dataset = load_from_disk("original_datasets/skypile_2023_sampled_100_eval")
# dataset = dataset.map(...)  # do all your processing here
dataset.push_to_hub("OrionZheng/skypile_2023_sampled_100_eval",
                    token="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",
                    private=True)