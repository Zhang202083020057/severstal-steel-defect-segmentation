# C2 类别定向增强实验

## 实验状态

- 状态：训练、离线推理及 Kaggle Code Submission 均已完成。
- 训练 Notebook：`zhanshuguo/severstal-c2-augmentation-train` Version 2。
- 离线提交 Notebook：`zhanshuguo/severstal-c2-code-submit` Version 1。
- Kaggle Submission Ref：`56189473`。

## 唯一实验变量

在 B0 的训练划分、模型、损失、输入尺寸、学习率、epoch、batch size、阈值和随机种子全部保持不变的前提下，只对训练集中包含 Class 2 的图片启用更强的在线增强。

- 非 C2 训练图：Resize、HorizontalFlip、Normalize。
- C2 训练图：额外加入小角度旋转、亮度/对比度、Gamma 和高斯噪声。
- 验证集：仅 Resize、Normalize，不做随机增强。
- 不生成新训练图片，不改变训练/验证划分，不进行过采样。

## 管线审计

队友提供的 `check_c2_pipeline.py` 已在完整原始 Severstal 数据上运行。本地生成的预览 PNG 和 JSON 报告位于 `pipeline_check/`，属于审计产物，不提交到 GitHub。

- 总图像：12,568。
- 训练/验证：10,054 / 2,514，与 B0 完全相同。
- Class 1-4 训练正样本：718 / 198 / 4,120 / 641。
- 训练验证重叠：0。
- C2/非 C2 transform 路由：通过。
- 几何变换图像与 mask 同步检查：IoU 1.0，通过。

## 公平对照命令

```bash
bash scripts/run_c2_kaggle.sh
```

与 `scripts/run_b0_kaggle.sh` 的结果比较时，应重点查看 Class 2 的正样本 Dice/召回，而不能只看包含大量空 mask 得分的四类平均 Dice。

## 训练与提交结果

最佳 checkpoint 为 epoch 10：

| 指标 | B0 | C2 定向增强 | 差值 |
|---|---:|---:|---:|
| Validation Dice | 0.906335 | **0.912265** | +0.005930 |
| Class 1 validation Dice | 0.928799 | 0.928799 | 0 |
| Class 2 validation Dice | 0.980509 | 0.980509 | 0 |
| Class 3 validation Dice | 0.769320 | **0.778168** | +0.008848 |
| Class 4 validation Dice | 0.946711 | **0.961584** | +0.014874 |
| Kaggle Public Dice | 0.86086 | **0.87369** | +0.01283 |
| Kaggle Private Dice | 0.85772 | **0.86944** | +0.01172 |

使用统一阈值 0.5 生成测试集预测时，各类别非空掩码数量为：Class 1 = 0、Class 2 = 0、Class 3 = 2,343、Class 4 = 291。

## 结论

这次单变量实验提升了总体验证分数以及 Kaggle Public/Private Dice，可以保留为当前整体成绩更好的版本。但它**没有证明 C2 检测能力得到提升**：Class 2 validation Dice 与 B0 完全相同，并且测试集仍没有任何非空 C2 预测。该 Dice 等于验证集中 C2 空掩码比例，说明阈值 0.5 下仍倾向于把 C2 全部预测为空。

排行榜提升主要伴随 Class 3/4 validation Dice 上升。可能原因是 C2 图片上的增强对模型产生了正则化作用；此外，包含 C2 的多标签图片会同步增强其他类别掩码。因此，文书中可以写“定向增强使整体 Kaggle Dice 提升”，暂时不能写“解决了 C2 类别不平衡”。下一步应先补充 C2 正样本 Dice、召回率与阈值搜索，再单独测试过采样或 Focal-Dice。

## 视觉检查提醒

当前 C2 旋转使用常量黑色边界。对细长钢板图像，即使旋转 2 度也会形成可见的黑色三角区域。第一轮应按队友原配置完成严格对照；如果 C2 没有提升，下一轮只把图像边界模式改为反射填充再验证，避免同时改变多个变量。
