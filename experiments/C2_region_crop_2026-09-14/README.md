# C2 region crop（待运行）

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
