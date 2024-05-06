import torch
from torch.nn import BCEWithLogitsLoss

# 假设我们有一组随机生成的 logits（通常是模型的输出）
logits = torch.randn(2, 5)  # 例如，模型为10个类别生成了一组 logits
print("原始 logits:\n", logits)

# 定义 top-k 的 k 值
k = 3

# 使用 torch.topk 获取 top-k logits 的索引
topk_values, topk_indices = torch.topk(logits, k, dim=-1)
print("Top-k 值:\n", topk_values)
print("Top-k 索引:\n", topk_indices)

# 创建一个与 logits 同形状的 mask，所有元素初始为一个非常小的值（比如负无穷）
mask = torch.zeros_like(logits)

# 只在 top-k 的位置上放置对应的原始 logits 值
mask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(mask))
print("生成的mask:\n", mask)


random_pred = torch.rand_like(mask)

print("Fake prediction:\n", random_pred)

bce_loss = BCEWithLogitsLoss()
loss = bce_loss(random_pred, mask)
print("BCE Loss:", loss.item())
