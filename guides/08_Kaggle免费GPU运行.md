# Kaggle 免费 GPU 运行指南

这份指南适用于 Kaggle Notebook，尤其是页面分配到 Tesla P100 时。

## 1. 添加比赛数据

在 Notebook 右侧点击 `Add Input`，搜索并添加官方比赛数据
`Severstal: Steel Defect Detection`。不要上传本地压缩包。

先查找真实路径：

```python
from pathlib import Path

csv_files = list(Path("/kaggle/input").rglob("train.csv"))
print(csv_files)
if not csv_files:
    raise RuntimeError("没有找到 train.csv，请先使用 Add Input 添加比赛数据")
DATA_DIR = csv_files[0].parent
print("DATA_DIR =", DATA_DIR)
print("train_images exists:", (DATA_DIR / "train_images").is_dir())
```

只有最后一行是 `True`，这个 `DATA_DIR` 才能交给训练程序。如果为 `False`，
查看打印出的路径和右侧 Input 目录，确认是否添加了非官方或不完整的数据集。

## 2. P100 与 PyTorch 兼容问题

P100 的计算能力是 `sm_60`。如果环境输出显示当前 PyTorch 只支持
`sm_70` 及以上，需要安装保留 Pascal 支持的 CUDA 12.6 wheel：

```bash
!pip install -q --force-reinstall \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

安装完成后重启 Notebook Session，然后验证真实 CUDA 计算，而不只是检查
`torch.cuda.is_available()`：

```python
import torch

print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_arch_list())
print(torch.ones(1, device="cuda") * 2)
```

输出的架构列表应包含 `sm_60`，最后一行应显示 CUDA tensor。

## 3. 安装项目依赖

```bash
!git clone https://github.com/Zhang202083020057/severstal-steel-defect-segmentation.git
%cd /kaggle/working/severstal-steel-defect-segmentation
!pip install -q -r requirements-kaggle.txt
```

如果仓库已经存在，不要再次 clone；进入原目录后执行 `!git pull` 即可。

## 4. 检查和冒烟测试

在 Python 单元中自动找到数据目录，再把路径传给命令：

```python
from pathlib import Path

csv_files = list(Path("/kaggle/input").rglob("train.csv"))
assert csv_files, "请先 Add Input"
DATA_DIR = csv_files[0].parent
assert (DATA_DIR / "train_images").is_dir(), DATA_DIR
```

```bash
!python check_environment.py --data-dir "{DATA_DIR}"
```

出现 `ENVIRONMENT CHECK PASSED` 后运行：

```bash
!python train.py \
  --data-dir "{DATA_DIR}" \
  --output-dir /kaggle/working/outputs/baseline_smoke \
  --architecture unet \
  --encoder resnet34 \
  --loss bce_dice \
  --augmentation basic \
  --smoke-test
```

冒烟测试成功后再开始正式训练，避免浪费 GPU 配额。
