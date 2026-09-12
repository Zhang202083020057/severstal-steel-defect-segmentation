# Severstal AutoDL 实验项目

这是面向初学者的可复现实验工程。完整学习顺序：

当前项目版本：**V2 / `v0.2.0`（C2 定向在线增强）**。版本改动、结果对照和复现入口见 [版本与实验进度](VERSION_HISTORY.md)。

1. [00_路线与模型选择](guides/00_路线与模型选择.md)
2. [01_AutoDL环境与数据上传](guides/01_AutoDL环境与数据上传.md)
3. [02_数据、Mask与RLE](guides/02_数据与RLE.md)
4. [03_跑通基线模型](guides/03_跑通基线.md)
5. [04_运行优化模型](guides/04_优化实验.md)
6. [05_阈值、TTA与提交](guides/05_推理与提交.md)
8. [07_Kaggle高分方案与选择依据](guides/07_Kaggle高分方案与选择依据.md)
9. [08_Kaggle免费GPU运行](guides/08_Kaggle免费GPU运行.md)
10. [实验记录表](guides/实验记录表.md)

当前已完成 B0 正式基线：U-Net + ResNet34，验证集最佳 Dice 为 `0.906335`；Kaggle 代码提交的 Public / Private Dice 分别为 `0.86086` / `0.85772`。
详细配置、逐轮结果、数据审计与类别不平衡解释、代码提交方式和后续实验计划见 [B0 基线实验记录](experiments/B0_baseline_2026-09-08.md)。

已完成 C2 定向增强对照：只对含 Class 2 的训练图增加轻微亮度/对比度、Gamma、高斯噪声和 ±2° 旋转，其余设置与 B0 相同且不使用过采样。Validation Dice 为 `0.912265`，Kaggle Public / Private Dice 为 `0.87369` / `0.86944`。整体分数上升，但阈值 0.5 下 C2 仍为全空预测，不能解释为 C2 检测问题已经解决。详见 [C2 定向增强实验记录](experiments/C2_augmentation_2026-09-12/README.md)。

| 版本 | 方案 | Validation Dice | Kaggle Public | Kaggle Private |
|---|---|---:|---:|---:|
| V1 | B0：基础增强 | 0.906335 | 0.86086 | 0.85772 |
| V2 | B0 + C2 定向在线增强，无过采样 | **0.912265** | **0.87369** | **0.86944** |

Kaggle 上复现 V1 或 V2 可分别运行：

```bash
bash scripts/run_b0_kaggle.sh
bash scripts/run_c2_kaggle.sh
```

## 目录

```text
autodl_project/
├── check_environment.py   # 检查GPU、依赖、数据和模型
├── visualize_samples.py   # 可视化原图与真实缺陷Mask
├── steel_common.py        # RLE、Dataset、模型、Loss、Dice
├── train.py               # 训练U-Net或DeepLabV3+
├── tune_thresholds.py     # 在验证集搜索四类阈值
├── predict.py             # TTA、后处理和提交文件生成
├── requirements.txt
├── requirements-kaggle.txt # Kaggle依赖（避免覆盖GPU对应的PyTorch）
├── scripts/                # 可直接运行的实验脚本
├── kaggle_code_submission/ # 离线可复现的 Kaggle 代码提交脚本与说明
├── guides/                # 逐步中文说明
└── experiments/           # 可复现实验配置与结果
```

## 最短执行路径

以下假设项目和数据都放在 `/root/autodl-tmp/kaggle/`：

```bash
cd /root/autodl-tmp/kaggle/autodl_project
pip install -r requirements.txt

python check_environment.py \
  --data-dir /root/autodl-tmp/kaggle/severstal-steel-defect-detection

python train.py \
  --data-dir /root/autodl-tmp/kaggle/severstal-steel-defect-detection \
  --output-dir /root/autodl-tmp/kaggle/outputs/baseline_smoke \
  --architecture unet \
  --smoke-test
```

看到 `saved new best model` 就表示端到端链路已经跑通。之后按照 guides 中的顺序进行正式实验。
