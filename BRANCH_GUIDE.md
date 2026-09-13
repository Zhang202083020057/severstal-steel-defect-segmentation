# V3 平衡采样实验分支查看指南

## 分支定位

- 当前分支：`experiment/c2-balanced-sampling`
- 直接对照分支：`experiment/c2-targeted-augmentation`（V2）
- 基线分支：`main`（V1 / B0）
- 实验目标：只改变训练图片的抽样方式，验证增加少数类进入 batch 的频率是否能改善 C2 检出能力。

查看 V3 相对 V2 的集中差异：

```text
https://github.com/Zhang202083020057/severstal-steel-defect-segmentation/compare/experiment/c2-targeted-augmentation...experiment/c2-balanced-sampling
```

## V3 改了什么

1. `train.py`
   - 新增 `--sampling {shuffle,sqrt_inverse_frequency}`。
   - 新增带上限的平方根反频率 `WeightedRandomSampler`。
   - 新增正样本 Dice、图片级检出召回率/精确率和预测非空率。
2. `steel_common.py`
   - 新增正样本 Dice 与图片级检出统计函数。
3. `scripts/run_c2_balanced_kaggle.sh`
   - 提供 V3 全量训练和测试推理入口。
4. `experiments/C2_balanced_sampling_2026-09-13/`
   - 保存实验设计、Kaggle 训练脚本、离线提交脚本和最终结果。

V3 继承 V2 的 C2 定向在线增强，只增加平衡采样。模型、split、seed、Loss、输入尺寸、batch size、epoch、学习率、数据增强、验证集处理与推理参数都保持不变。

## 如何复现

```bash
git switch experiment/c2-balanced-sampling
bash scripts/run_c2_balanced_kaggle.sh
```

默认数据目录为：

```text
/kaggle/input/competitions/severstal-steel-defect-detection
```

数据集、模型权重和 `submission.csv` 不保存在 GitHub，需要在 Kaggle 中挂载官方比赛数据。

## 三个版本结果

| 指标 | V1：B0 | V2：C2-AUG | V3：C2-AUG + 平衡采样 |
|---|---:|---:|---:|
| Validation Dice | 0.906335 | **0.912265** | 0.910129 |
| Kaggle Public Dice | 0.86086 | **0.87369** | 0.86040 |
| Kaggle Private Dice | 0.85772 | **0.86944** | 0.86793 |

- V3 Submission Ref：`56199102`。
- 训练 Notebook：`zhanshuguo/severstal-c2-balanced-sampling-train` Version 1。
- 提交 Notebook：`zhanshuguo/severstal-c2-balanced-code-submit` Version 1。

## 结论边界

V3 没有超过 V2。虽然 C2 每个 epoch 的期望抽取次数由 198 增加到约 715，但 0.5 阈值下的 C2 正样本 Dice、检出召回率和测试集非空预测数仍均为 0。因此本实验支持的结论是“当前平衡采样策略无效”，不能写成“解决了类别不平衡”或“提升了 C2 检测”。

完整设计与逐类诊断见 [`experiments/C2_balanced_sampling_2026-09-13/README.md`](experiments/C2_balanced_sampling_2026-09-13/README.md)。
