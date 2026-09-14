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

## V3｜C2 定向增强 + 平方根反频率平衡采样

- 分支：`experiment/c2-balanced-sampling`。
- 以 V2 为对照，只增加有放回平衡采样；C2 每个 epoch 的期望抽取次数由 198 增加到约 715。
- Validation Dice：0.910129。
- Kaggle Public / Private Dice：0.86040 / 0.86793。
- C2 正样本 Dice 和检出召回率仍为 0，因此没有解决 C2 检出问题，也没有超过 V2。
- Submission Ref：56199102。
- 详细记录位于对应实验分支的 `experiments/C2_balanced_sampling_2026-09-13/README.md`。

## V4｜V2 权重 + 分类别阈值与面积校准

- 分支：`experiment/c2-threshold-calibration`。
- 不重新训练，直接复用 V2 权重。
- 概率诊断确认 C2 通道塌缩：C2 正样本的全图最大概率中位数仅约 4.77e-7，阈值降至 0.01 仍无检出。
- 最终阈值：`0.5,0.5,0.5,0.075`。
- 原图空间最小总面积：`0,0,800,800`。
- 校准后 Validation Dice：0.917863。
- Kaggle Public / Private Dice：0.88093 / 0.87886。
- 相对 V2：Public +0.00724，Private +0.00942。
- Submission Ref：56203996。
- 结论：V4 是当前整体最佳版本，但提升来自 C3/C4 后处理，不是 C2。
- 详细记录：[C2 概率诊断与阈值校准](experiments/C2_threshold_calibration_2026-09-13/README.md)。

## V5a｜C2 标签与缩放审计

- 分支：`experiment/c2-resolution-audit`。
- 不训练模型，检查 247 张 C2 mask 从 256×1600 最近邻缩放到 256×800 后的像素与连通区域变化。
- 缩放后完全消失：0 张。
- 扣除宽度减半预期面积变化后的像素保留率：中位数 1.0000，最小值 0.9367。
- 仅 3 张图片出现连通区域减少。
- 结论：当前缩放不是 C2 通道塌缩的主要原因；下一步优先测试 Focal-Dice。
- 详细记录：[C2 标签与缩放审计](experiments/C2_resolution_audit_2026-09-13/README.md)。

## V5b｜C2 定向增强 + Focal-Dice

- 分支：`experiment/c2-focal-dice`。
- 以 V2 为对照，只把 `BCE+Dice` 改为 `Focal+Dice`，其余训练配置不变。
- Validation Dice：0.905792。
- C2 真实区域最大概率中位数仍为 0，C2 正样本 Dice 和检出 F1 仍为 0。
- Kaggle Public / Private Dice：0.86886 / 0.87045。
- Submission Ref：56207881。
- 结论：Focal-Dice 单独没有激活 C2 输出通道，也没有超过 V4。
- 详细记录：[C2 Focal-Dice 实验](experiments/C2_focal_dice_2026-09-13/README.md)。

## V6｜C2 定向增强 + C2 region crop

- 分支：`experiment/c2-region-crop`。
- 以 V2 为基础，仅对含 C2 的训练图片截取一个保证包含 C2 的 800 像素宽窗口，再缩放到 256×800。
- 验证集和测试集仍使用完整图片。
- 最佳 Validation Dice：0.911102。
- 测试集非空预测：C1=0、C2=0、C3=1777、C4=308。
- C2 仍为全空预测，说明 region crop 单独没有解决 C2 通道塌缩。
- Submission Ref：56235278（等待排行榜评分）。
- 详细记录：[C2 region crop 实验](experiments/C2_region_crop_2026-09-14/README.md)。

## V1～V4 排行榜对比

| 指标 | V1：B0 | V2：C2-AUG | V3：平衡采样 | V4：校准后处理 |
|---|---:|---:|---:|---:|
| Validation Dice | 0.906335 | 0.912265 | 0.910129 | **0.917863** |
| Kaggle Public Dice | 0.86086 | 0.87369 | 0.86040 | **0.88093** |
| Kaggle Private Dice | 0.85772 | 0.86944 | 0.86793 | **0.87886** |

V4 是当前整体榜单成绩最好的版本，但 C1/C2 仍为全空预测。概率诊断已排除“只需降低 C2 阈值”的假设；下一版应修改训练优化目标，优先单变量比较 Focal-Dice，并保留 V4 后处理用于最终推理。
