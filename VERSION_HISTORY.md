# 版本与实验进度

本项目采用“一个版本只验证一组明确改动”的方式维护，确保每次榜单变化都能追溯到代码、参数和实验记录。数据集、模型权重与 `submission.csv` 不提交到 GitHub。

## V1｜B0 基线

- 状态：已完成训练、测试推理和 Kaggle Code Submission。
- 模型：U-Net + ResNet34（ImageNet 预训练）。
- 输入：256×800，batch size 4，10 epochs，seed 42。
- Loss：BCE + Dice。
- 训练增强：Resize、随机水平翻转、Normalize。
- 推理：四类阈值均为 0.5，无 TTA、无面积后处理。
- Validation Dice：0.906335。
- Kaggle Public / Private Dice：0.86086 / 0.85772。
- Submission Ref：56126689。
- 详细记录：[B0 基线实验](experiments/B0_baseline_2026-09-08.md)。

复现命令：

```bash
bash scripts/run_b0_kaggle.sh
```

## V2｜C2 定向在线增强

- 状态：已完成训练、测试推理和 Kaggle Code Submission。
- 对照原则：模型、split、seed、Loss、输入尺寸、batch size、epoch、非 C2 增强、验证集处理与推理参数均保持 V1 不变。
- 唯一训练变量：含 Class 2 的训练图片额外随机使用轻微亮度/对比度、Gamma、高斯噪声和 ±2° 旋转。
- 不生成新的训练数据文件，不进行过采样；C2 训练图片仍为 198 张/epoch。
- Validation Dice：0.912265（相对 V1 +0.005930）。
- Kaggle Public / Private Dice：0.87369 / 0.86944（相对 V1 +0.01283 / +0.01172）。
- Submission Ref：56189473。
- 训练 Notebook：`zhanshuguo/severstal-c2-augmentation-train` Version 2。
- 离线提交 Notebook：`zhanshuguo/severstal-c2-code-submit` Version 1。
- 详细记录：[C2 定向增强实验](experiments/C2_augmentation_2026-09-12/README.md)。

复现命令：

```bash
bash scripts/run_c2_kaggle.sh
```

## V1 与 V2 对比

| 指标 | V1：B0 | V2：C2-AUG | 差值 |
|---|---:|---:|---:|
| Validation Dice | 0.906335 | **0.912265** | +0.005930 |
| Kaggle Public Dice | 0.86086 | **0.87369** | +0.01283 |
| Kaggle Private Dice | 0.85772 | **0.86944** | +0.01172 |

V2 是当前整体榜单成绩更好的版本，但不能声称已经解决 C2 检测问题。统一阈值 0.5 下，V2 的测试集 C2 非空预测仍为 0；Validation Dice 的提升主要来自 Class 3 和 Class 4。下一版本应补充 C2 正样本 Dice/召回率与独立阈值搜索，再单独验证过采样或 Focal-Dice。
