# V4 阈值校准实验分支查看指南

## 分支定位

- 当前分支：`experiment/c2-threshold-calibration`
- 直接对照：`experiment/c2-targeted-augmentation`（V2）
- 实验目标：不重新训练，诊断 C2 概率并校准四类阈值和最小预测面积。

GitHub 差异页：

```text
https://github.com/Zhang202083020057/severstal-steel-defect-segmentation/compare/experiment/c2-targeted-augmentation...experiment/c2-threshold-calibration
```

## 主要改动

1. `tune_thresholds.py`
   - 同时搜索分类别阈值与最小总面积；
   - 新增正样本 Dice、检出召回率/精确率/F1；
   - 新增正负样本及真实缺陷区域概率分布。
2. `experiments/C2_threshold_calibration_2026-09-13/kaggle_calibration/`
   - Kaggle GPU 概率诊断 Notebook。
3. `experiments/C2_threshold_calibration_2026-09-13/kaggle_code_submit/`
   - 使用 V2 权重和校准参数的离线 Code Submission。

## 结果

- C2 正样本全图最大概率中位数约 `4.77e-7`；真实 C2 区域最大概率中位数为 `0`。
- 阈值降至 0.01 后，C2 正样本 Dice 和检出召回率仍为 0。
- 最终推理阈值：`0.5,0.5,0.5,0.075`。
- 原图空间最小总面积：`0,0,800,800`。
- Validation Dice：`0.917863`。
- Kaggle Public / Private Dice：`0.88093 / 0.87886`。
- Submission Ref：`56203996`。

V4 是当前整体最佳版本，但提升来自 C3/C4 后处理，C2 问题仍未解决。

完整记录见 [`experiments/C2_threshold_calibration_2026-09-13/README.md`](experiments/C2_threshold_calibration_2026-09-13/README.md)。
