# C2 定向增强分支查看指南

## 分支定位

- 分支：`experiment/c2-targeted-augmentation`
- 对照基线：`main`（V1 / B0）
- 实验版本标签：`v0.2.0`
- 实验目标：只改变含 Class 2 图片的在线数据增强，验证其相对 B0 的影响。

GitHub 可通过下面的比较页集中查看本分支相对 `main` 的代码和文档差异：

```text
https://github.com/Zhang202083020057/severstal-steel-defect-segmentation/compare/main...experiment/c2-targeted-augmentation
```

## 改了什么

1. `steel_common.py`
   - 新增 `build_c2_transforms()`。
   - 当训练图片包含 Class 2 时，使用 C2 专用 transform。
   - 图像和四通道 mask 同步执行几何变换。
2. `train.py`
   - 新增命令行参数 `--c2-augmentation`。
   - 未开启该参数时，行为与 B0 保持一致。
3. `check_c2_pipeline.py`
   - 核对数据划分、类别数量、C2/非 C2 路由及图像-mask 同步性。
4. `scripts/run_c2_kaggle.sh`
   - 提供 V2 全量训练和测试推理的一键命令。
5. `experiments/C2_augmentation_2026-09-12/`
   - 保存实验说明、Kaggle 训练脚本和离线 Code Submission 脚本。

## 唯一实验变量

V2 对含 C2 的训练图片额外随机使用：

- 亮度/对比度调整；
- Gamma 调整；
- Gaussian Noise；
- ±2° 旋转。

以下内容均与 B0 保持一致：U-Net + ResNet34、ImageNet 预训练、BCE+Dice、训练/验证划分、seed 42、256×800、batch size 4、10 epochs、学习率、非 C2 增强、验证集处理、0.5 推理阈值、无 TTA 和无面积后处理。

该方案是**在线定向增强**，不是过采样：每个 epoch 中仍只有 198 张 C2 训练图片，不生成新图片文件，也不增加 C2 进入 batch 的次数。

## 如何复现

```bash
git switch experiment/c2-targeted-augmentation
bash scripts/run_c2_kaggle.sh
```

默认数据目录为：

```text
/kaggle/input/competitions/severstal-steel-defect-detection
```

数据集、模型权重和 `submission.csv` 不保存在 GitHub，需要在 Kaggle 中挂载官方比赛数据。

## 已完成结果

| 指标 | main / B0 | 本分支 / C2-AUG | 差值 |
|---|---:|---:|---:|
| Validation Dice | 0.906335 | **0.912265** | +0.005930 |
| Kaggle Public Dice | 0.86086 | **0.87369** | +0.01283 |
| Kaggle Private Dice | 0.85772 | **0.86944** | +0.01172 |

- Kaggle Submission Ref：`56189473`。
- 训练 Notebook：`zhanshuguo/severstal-c2-augmentation-train` Version 2。
- 提交 Notebook：`zhanshuguo/severstal-c2-code-submit` Version 1。

## 结果边界

整体 Kaggle Dice 得到了提升，但统一阈值 0.5 下，测试集 Class 2 非空预测仍为 0；Class 2 validation Dice 与 B0 相同，并且等于验证集中该类空 mask 的比例。因此当前证据只支持“定向增强提高了整体分数”，不支持“已经提升 C2 检出能力”或“已经解决类别不平衡”。

更完整的逐类结果和原因分析见 [`experiments/C2_augmentation_2026-09-12/README.md`](experiments/C2_augmentation_2026-09-12/README.md)。
