# C2 region crop（已训练并提交）

## 目的

仅对训练集中含 C2 的图片，随机截取一个包含 C2 缺陷的水平窗口，再缩放到 `256×800`。这样能提高 C2 在输入中的像素占比，验证集和测试集仍使用完整原图。

## 控制变量

- 模型：U-Net + ResNet34 ImageNet
- 损失：BCE-Dice
- 训练：10 epochs，batch size 4，learning rate 3e-4
- 其他设置：与 B0/C2 定向增强保持一致
- 新变量：`--c2-crop --c2-crop-width 800`

## 判断标准

重点记录：总体验证 Dice、C2 Dice、C2 recall、C2 非空预测数量，以及 Kaggle Public/Private 分数。若 C2 仍为 0，说明问题不只是缺陷在整图中占比太小，需要转向 C2 专用模型或图像级分类损失。

## Kaggle 运行

运行 `kaggle_train/c2_crop_train.py`。完成后下载 `best_model.pt`、`history.csv` 和 `config.json`，再使用现有 `predict.py` 推理提交。

## 实验结果

- 训练 Kernel：`zhanshuguo/severstal-c2-region-crop-train` Version 3。
- 最佳 Validation Dice：`0.911102`（epoch 10）。
- 日志中的 Class 2 Dice：`0.980509`，但该指标被大量 C2 空样本主导，不能视为 C2 正样本检出成功。
- 测试集非空预测：C1=0、C2=0、C3=1777、C4=308。
- 推理参数：thresholds=`0.5,0.5,0.5,0.075`，min total pixels=`0,0,800,800`。
- Submission Ref：`56235278`（等待评分）。

## 当前结论

C2 region crop 改善了训练时缺陷相对面积，但测试集 C2 仍为全空预测，因此它单独无法解决 C2 通道塌缩。下一步不应继续只调整 crop/阈值，应该考虑 C2 图像级存在性损失或独立 C2 二分类分割模型。
