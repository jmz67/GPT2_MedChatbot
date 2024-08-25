import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=1.0):
        """
        初始化LoRA层
        ：param in_features: 输入特征的维度
        :param out_features: 输出特征的维度
        :param r: 低秩矩阵的秩
        :param alpha: LoRA的缩放系数
        """
        super(LoRALayer, self).__init__()
        self.r = r
        self.alpha = alpha

        # 初始化A和B矩阵，分别为(in_features, r) 和 (r, out_features)
        self.A = nn.Parameter(torch.randn(out_features, r) * 0.01)
        self.B = nn.Parameter(torch.randn(r, out_features) * 0.01)

        # 可选的，冷冻原始权重矩阵 W
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

    def forward(self, x):
        # 执行原始线性层计算并加上低秩矩阵的结果
        return x @ self.weight.T + (x @ self.A) @ self.B.T * self.alpha
