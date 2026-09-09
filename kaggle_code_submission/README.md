# Kaggle 离线代码提交

该目录记录 B0 基线实际使用的 Kaggle 代码提交方案。它解决了 Kaggle 保存 Notebook 后的断网重跑环境无法在线安装 `segmentation-models-pytorch` 的问题。

## 提交前的私有输入

不提交模型、预测文件或二进制 wheel 到 GitHub。Kaggle Notebook 需要附加以下私有输入：

1. 比赛数据：`severstal-steel-defect-detection`；
2. `zhanshuguo/severstal-b0-assets`：`best_model.pt`、`predict.py`、`steel_common.py`；
3. `zhanshuguo/severstal-b0-offline-wheels`：Linux/CPython 3.12 对应的 `Pillow==11.3.0`、`timm==1.0.29`、`segmentation-models-pytorch==0.5.0` wheel。

`kernel-metadata.json` 指定 T4 GPU 而非 P100：Kaggle 默认 PyTorch 不支持 P100 的 Pascal `sm_60` 架构；T4 可直接使用默认 PyTorch。

## 运行与提交

在已配置 Kaggle API 凭据的本机运行：

```powershell
py -m kaggle kernels push -p kaggle_code_submission --accelerator NvidiaTeslaT4

py -m kaggle competitions submit severstal-steel-defect-detection `
  -k zhanshuguo/severstal-b0-code-submit `
  -v 1 `
  -f submission.csv `
  -m "B0 baseline: U-Net ResNet34, BCE+Dice, 256x800, 10 epochs"
```

Version 1 已完成：提交编号 `56126689`，Public Dice `0.86086`，Private Dice `0.85772`。

## 输出检查

运行成功时，日志应包含：

```text
wrote 22024 rows to /kaggle/working/submission.csv
SUBMISSION_READY: /kaggle/working/submission.csv
```

输出文件应含有 `ImageId_ClassId`、`EncodedPixels` 两列以及 22,024 行。
