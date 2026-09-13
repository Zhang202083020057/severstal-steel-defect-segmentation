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

- 当前状态：Kaggle 全量训练、测试推理、Code Submission 和排行榜评分均已完成。
- 最佳 epoch：8。
- Validation Dice：`0.910129`，比 V2 的 `0.912265` 下降 `0.002136`。
- Kaggle Public / Private Dice：`0.86040 / 0.86793`。
- 相对 V2：Public `-0.01329`，Private `-0.00151`。
- 相对 B0：Public `-0.00046`，Private `+0.01021`。
- Kaggle Submission Ref：`56199102`。
- 训练 Notebook：`zhanshuguo/severstal-c2-balanced-sampling-train` Version 1。
- 离线提交 Notebook：`zhanshuguo/severstal-c2-balanced-code-submit` Version 1。

### 最佳 epoch 的诊断结果

| 类别 | 常规 Validation Dice | 正样本 Dice | 检出召回率 | 检出精确率 | 预测非空率 |
|---|---:|---:|---:|---:|---:|
| C1 | 0.928799 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| C2 | 0.980509 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| C3 | 0.779217 | 0.586360 | 0.907767 | 0.878759 | 0.423230 |
| C4 | 0.951991 | 0.351905 | 0.693750 | 0.867188 | 0.050915 |

测试集共生成 `22,024` 行（5,506 张图片 × 4 类）。阈值 0.5 下，各类非空预测数为 C1=`0`、C2=`0`、C3=`2,034`、C4=`239`。

### 阶段结论

平衡采样把 C2 每个 epoch 的期望抽取次数从 198 增加到约 715，但 C2 的正样本 Dice 和检出召回率仍为 0，整体 Validation Dice 与 Kaggle Public/Private 分数也都低于 V2。因此当前证据表明：在保持 BCE+Dice、统一 0.5 阈值及其他训练参数不变时，这种保守的平方根反频率过采样没有解决 C2 检出问题，也没有带来稳定的整体收益。

下一步不应继续盲目提高 C2 重复采样倍数，应优先在验证集保存概率并做分类别阈值搜索，同时检查 C2 输出概率分布；如果 C2 概率整体偏低，再对比 Focal-Dice 或类别加权 BCE。
