# Severstal AutoDL 实验项目

这是面向初学者的可复现实验工程。完整学习顺序：

1. [00_路线与模型选择](guides/00_路线与模型选择.md)
2. [01_AutoDL环境与数据上传](guides/01_AutoDL环境与数据上传.md)
3. [02_数据、Mask与RLE](guides/02_数据与RLE.md)
4. [03_跑通基线模型](guides/03_跑通基线.md)
5. [04_运行优化模型](guides/04_优化实验.md)
6. [05_阈值、TTA与提交](guides/05_推理与提交.md)
7. [06_结果分析与文书](guides/06_结果分析与文书.md)
8. [07_Kaggle高分方案与选择依据](guides/07_Kaggle高分方案与选择依据.md)
9. [08_Kaggle免费GPU运行](guides/08_Kaggle免费GPU运行.md)
10. [实验记录表](guides/实验记录表.md)

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
└── guides/                # 逐步中文说明
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
