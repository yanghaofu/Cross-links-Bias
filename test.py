from ablation import *

# 初始化平均融合模块
average_merger = AverageMerger()

# 给定两个嵌入向量 emb_ori 和 emb_aug
emb_ori = torch.randn(10, 128)  # 示例原始嵌入
emb_aug = torch.randn(10, 128)  # 示例增强嵌入

# 计算平均融合后的嵌入
merged_embedding = average_merger(emb_ori, emb_aug)
print(merged_embedding)