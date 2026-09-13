# V3｜C2 定向增强 + 平衡采样

## 实验目标

在 V2 的 C2 定向在线增强基础上，只改变训练图片的抽样方式，验证“让少数类更频繁进入 batch”是否能改善 C2 检出能力。

## 唯一实验变量

- V2：普通随机打乱，每个 epoch 每张训练图片读取一次。
- V3：`WeightedRandomSampler` 有放回抽样，每个 epoch 仍抽取 10,054 张，但稀有类别图片获得更高概率。

类别权重根据训练集正样本数量动态计算：

```text
weight(class) = sqrt(max_class_count / class_count)
```

权重限制在 1～5 倍；多标签图片采用其所含类别的最大权重，避免权重简单相加。该设计比直接按 C3/C2≈20.8 倍过采样更保守，可降低 C2 只有 198 张时的过拟合风险。

按固定训练划分计算，Class 1～4 的采样权重约为 `2.395 / 4.562 / 1.000 / 2.535`。预计每个 epoch 抽到的 C2 图片由 198 次增加至约 715 次；batch size 4 时完全没有 C2 的近似概率由 92.4% 降至 74.4%。这是保守的第一轮采样实验，并非强制每个 batch 都含 C2。

以下设置与 V2 保持一致：数据划分、seed 42、U-Net + ResNet34、ImageNet 预训练、BCE+Dice、C2 定向增强、256×800、batch size 4、10 epochs、学习率 0.0003、阈值 0.5、无 TTA 和无面积后处理。

## 新增诊断指标

除原有包含空掩码的分类别 Dice 外，本版本额外记录：

- `val_positive_dice_class_*`：只在真实 mask 非空的图片上计算 Dice；
- `val_detection_recall_class_*`：真实含该类的图片中，有多少被预测为非空；
- `val_detection_precision_class_*`：预测非空的图片中，有多少真实含该类；
- `val_predicted_nonempty_rate_class_*`：验证集中预测为非空的比例。

这些指标用于判断 C2 是否真正被检出，不再把空 mask 的 Dice=1 当作少数类效果提升。

## 运行

```bash
bash scripts/run_c2_balanced_kaggle.sh
```

## 状态

- 当前状态：代码检查完成，等待/正在进行 Kaggle 全量训练。
- 结果将在训练与 Code Submission 完成后补充。
