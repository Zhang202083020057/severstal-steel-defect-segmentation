# 01｜AutoDL 环境与数据上传

## 第一步：创建实例

选择带 CUDA 的 PyTorch 镜像。入门实验建议：

- GPU 显存至少 16 GB；
- 系统盘只存环境；
- 数据集、权重和输出放在 `/root/autodl-tmp` 持久化数据盘；
- 训练结束后及时关机，避免继续计费。

## 第二步：上传文件

将以下内容上传到：

```text
/root/autodl-tmp/kaggle/
├── autodl_project/
└── severstal-steel-defect-detection/
    ├── train.csv
    ├── sample_submission.csv
    ├── train_images/
    └── test_images/
```

不要上传本地的 `runs/smoke`。原始 ZIP 上传后可在 AutoDL 解压，但确认解压正确后不要重复保留不必要副本。

## 第三步：安装依赖

```bash
cd /root/autodl-tmp/kaggle/autodl_project
pip install -r requirements.txt
```

如果安装导致 PyTorch 被替换，先停止，不要连续尝试多个 CUDA 版本。保存完整终端输出后检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

第一次正式训练时，ImageNet 预训练权重可能需要联网下载。如果下载失败，先保留完整错误信息；临时使用 `--encoder-weights none` 虽然可以训练，但不能和“预训练 ResNet34”实验混为一谈。

## 第四步：环境总检查

```bash
python check_environment.py \
  --data-dir /root/autodl-tmp/kaggle/severstal-steel-defect-detection
```

程序将检查：

- CUDA 和 GPU 名称；
- 关键 Python 包版本；
- 训练图是否为 12568 张；
- 测试图是否为 5506 张；
- 四类正掩码数量；
- RLE 编解码能否还原；
- U-Net 是否能输出 `1×4×64×400` 张量。

只有看到：

```text
ENVIRONMENT CHECK PASSED
```

再开始训练。

## 第五步：先可视化标签

```bash
python visualize_samples.py \
  --data-dir /root/autodl-tmp/kaggle/severstal-steel-defect-detection \
  --output-dir /root/autodl-tmp/kaggle/outputs/sample_overlays \
  --count 8
```

下载几张生成的 PNG，确认彩色 Mask 确实覆盖在钢板缺陷上。这一步用于发现 RLE 方向错误，不能跳过。

## 常用排错命令

```bash
nvidia-smi
df -h
free -h
pwd
find /root/autodl-tmp/kaggle -maxdepth 2 -type f | head
```

- `nvidia-smi`：GPU、显存和进程；
- `df -h`：磁盘空间；
- `free -h`：内存；
- `pwd`：当前目录；
- `find`：检查上传路径。
