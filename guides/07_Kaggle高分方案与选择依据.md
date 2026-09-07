# 07｜Kaggle 高分方案与我们的选择依据

本文只把高分方案当作研究参考，不复制其最终集成。我们的目标是在有限时间内完成一套能复现、能对比、能在面试中解释的单模型实验。

## 参考 1：冠军方案

[Kaggle 1st Place Solution](https://www.kaggle.com/competitions/severstal-steel-defect-detection/writeups/1st-place-solution)

冠军方案的核心包括：

- 分割模型主要为 U-Net 和 FPN，编码器采用 EfficientNet-B3；
- 分类器先过滤大约一半无缺陷图片，降低空图误报；
- 训练使用 `256×512` crop；
- 使用水平/垂直翻转、亮度对比度增强；
- 使用加权 BCE，以及 `0.75 BCE + 0.25 Dice`；
- 每类分别设置标签阈值和像素阈值；
- 预测总像素小于 `(600,600,900,2000)` 时清空该类 Mask，并删除小于 150 像素的组件；
- 最终依赖多个模型集成和两轮伪标签。

冠军公开榜约 0.92124、私榜约 0.90883。这里最大的启发不是照抄 9 个模型，而是：**空图过滤、面积阈值和后处理对该 Dice 指标非常重要**。

## 参考 2：第三名方案

[Kaggle 3rd Place Solution](https://www.kaggle.com/competitions/severstal-steel-defect-detection/writeups/chienyichi-3rd-place-solution)

第三名使用：

- U-Net 和 FPN；
- EfficientNet-B3/B4/B5、SE-ResNeXt50 编码器；
- Focal Loss 与 Weighted Sampler；
- `256×800` 训练、`256×1600` 推理；
- 水平和垂直翻转增强；
- 9 个模型平均，但没有使用 TTA；
- 分类别标签阈值和统一像素阈值。

其最佳私榜约 0.90934。这支持我们比较 Focal Loss、采用 `256×800` 起步，以及将 FPN 纳入候选；但 9 模型集成超出当前申请项目的必要范围。

## 参考 3：第四名方案

[Kaggle 4th Place Solution](https://www.kaggle.com/competitions/severstal-steel-defect-detection/writeups/ods-ai-wonderbolts-4th-place-solution)

第四名的重要观察：

- 重型编码器与简单 ResNet34 的效果接近；
- 最终仍以 U-Net/FPN 为主；
- 使用多模型、多折和不同随机种子增加集成多样性；
- 使用翻转 TTA、0.55 二值阈值及最小 256 像素过滤；
- 分类器/soft gating 负责处理空 Mask。

这给出选择 ResNet34 的直接理由：它不是为了“模型简单”随便选的，而是在该比赛中具有较好的效果—成本平衡。

## 参考 4：接近冠军私榜的完整开源代码

[GitHub: khornlund/severstal-steel-defect-detection](https://github.com/khornlund/severstal-steel-defect-detection)

该作者的最佳提交私榜约 0.91023，并公开了完整训练代码。主要结论：

- 使用 SMP 框架的 U-Net 和 FPN；
- EfficientNet-B5、SE-ResNeXt50 等编码器；
- 使用 `0.6 BCE + 0.4 Dice`；
- 训练分辨率宽度约 384～480，更大尺寸与更大 batch 都可能有益；
- 使用每类预测总像素阈值 `(600,600,1000,2000)`；
- 分类器在私榜明显减少误报；
- 作者尝试 DeepLabV3 后没有得到好结果；
- 作者认为水平翻转 TTA 收益不明显且增加推理耗时。

这直接说明：**DeepLabV3+有合理理论动机，但没有这场比赛的高分经验保证**。我们必须把它作为候选实验，而不是先写成“优化模型”。

## 参考 5：第十名方案的反例

[Kaggle 10th Place Solution](https://www.kaggle.com/competitions/severstal-steel-defect-detection/writeups/ods-ai-alexey-rozhkov-10th-place-solution)

第十名主要使用 FPN，并报告：

- 默认设置下 U-Net、PSPNet 不如其 FPN；
- 无增强在一些小模型实验中反而优于常规增强；
- 即使水平翻转理论上合理，也不能假设所有通用增强一定有效。

这提醒我们：数据增强必须通过同一验证集的对照实验验证，不能因为老师模板写了“数据增强策略”就直接声称提升。

## 综合结论

### 为什么 B0 使用 U-Net + ResNet34？

1. U-Net 是成熟且易解释的分割基线；
2. 多个前列方案实际采用 U-Net；
3. ResNet34 有 ImageNet 预训练权重，训练成本适中；
4. 第四名明确指出重编码器未必优于简单 ResNet34；
5. 它方便与 FPN、DeepLabV3+共用编码器做公平比较。

### 为什么加入 FPN？

U-Net 和 FPN 是本竞赛前列方案最常出现的两个解码框架，FPN 融合多个尺度的编码器特征，具有直接的竞赛经验支持。

### 为什么仍然测试 DeepLabV3+？

1. ASPP 理论上适合尺度差异明显的缺陷；
2. 老师的文书模板明确提出这一框架；
3. 负面公开结果使它成为有价值的研究问题：我们可以验证其在当前设置中是否真的不如 U-Net/FPN。

但如果 DeepLabV3+没有胜出，最终文书必须改成真实最佳模型，不能为了匹配模板隐瞒实验结果。

### 为什么比较 Focal Loss？

第三名使用 Focal Loss，且本数据存在类别与像素双重不平衡；但冠军使用加权 BCE/BCE-Dice，因此 Focal 是有依据的候选，而不是唯一正确答案。

### 为什么后处理优先级高？

冠军和多个前列方案都使用标签/像素阈值、整张 Mask 面积阈值、组件过滤或分类器 gating。由于空 Mask 误报的 Dice 代价很大，这些方法与本竞赛指标直接相关，通常比盲目换更重网络更值得优先验证。

### 为什么不立刻复制冠军完整方案？

冠军使用多模型集成、分类器和伪标签，训练及调参成本高，也很难在短时间内真正理解。申请项目先完成三个单模型公平对照、一个 Loss 实验和一个后处理实验，更容易证明独立分析能力。

## 最终决策规则

1. B0、A1、A2 在同一验证集比较；
2. 选平均 Dice 最高且四类结果合理的架构；
3. 在该架构上单独比较 Focal-Dice；
4. 再单独比较强增强；
5. 最后比较阈值、面积后处理和 TTA；
6. 只将真实有效的策略写进最终文书。

