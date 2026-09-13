# V5b｜C2 定向增强 + Focal-Dice

## 实验目标

V5a 已排除 256×1600 缩放至 256×800 导致 C2 mask 普遍消失的假设。本实验以 V2 为直接对照，只把 `BCE+Dice` 改为 `Focal+Dice`，验证降低大量易分类背景像素的影响后，C2 通道是否能产生有效正预测。

## 单一变量

| 配置 | V2 | V5b |
|---|---|---|
| Loss | BCE + Dice | Focal + Dice |
| 模型 | U-Net + ResNet34 | 不变 |
| ImageNet 预训练 | 是 | 不变 |
| C2 定向增强 | 是 | 不变 |
| 采样 | 普通 shuffle | 不变 |
| 输入 | 256×800 | 不变 |
| split / seed | 0.2 / 42 | 不变 |
| batch / epochs / lr | 4 / 10 / 0.0003 | 不变 |

当前 `Focal Loss` 使用 `alpha=0.75`、`gamma=2.0`，四个输出通道使用同一组参数。本轮暂不加入额外 C2 类别权重，避免同时改变多个因素。

## 评估标准

除整体 Validation Dice 和 Kaggle Dice 外，重点检查：

- C2 正样本 Dice；
- C2 图片级检出召回率与精确率；
- C2 正样本真实区域最大概率；
- 0.5 阈值及校准阈值下的 C2 非空预测数量。

只有 C2 正样本指标从 0 提高，才能认为 Focal-Dice 对 C2 有效。

## 状态

- 当前状态：训练、概率诊断和离线 Code Submission 已完成；竞赛提交 `56207881` 已被接收，当前仍处于 `PENDING` 评分状态。

## 已完成结果

- Training Notebook：`zhanshuguo/severstal-c2-focal-dice-train` Version 2。
- Validation Dice：`0.905792`（最佳 epoch 10），低于 V2 的 `0.912265`。
- 默认 0.5 阈值下测试集非空预测：C1=`0`、C2=`0`、C3=`2,468`、C4=`177`。
- Focal-Dice 概率诊断显示，C2 正样本真实区域最大概率中位数仍为 `0`；C2 正样本 Dice 与检出 F1 仍为 `0`。
- 校准提交使用阈值 `0.5,0.5,0.3,0.01`、原图空间面积门槛 `0,0,800,800`。
- 提交 Notebook：`zhanshuguo/severstal-c2-focal-submit` Version 1。

### 当前判断

Focal-Dice 没有激活 C2 输出通道，且整体验证 Dice 下降。因此仅改变 Loss 仍不足以解决 C2；下一步应检查训练标签的类别通道、C2 的像素级分布和模型结构/输出头，并考虑将 C2 作为独立的辅助分类任务或使用 C2 缺陷 patch 训练。
