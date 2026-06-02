# DermFM-Zero 复现踩坑记录

本文档记录以审稿人视角，从干净环境开始按 README 步骤复现 zero-shot 实验、linear-probing 实验、以及 `examples/zero-shot-classification.ipynb` 时遇到的问题。

环境：
- 主机：Linux 6.14.0 / aarch64
- GPU：NVIDIA GB10（Grace Blackwell，sm_121）
- Driver：580.126.09 / CUDA 13.0
- 工作目录：`/home/xieji/repos/DermFM-Zero`

---

## 0. 模型权重可见性

README 在顶部声明：

> 🔒 The DermFM-Zero model weights are private at this stage and available only upon reasonable request to the corresponding author … released upon publication.

但 README 给出的所有脚本（`--model 'hf-hub:redlessone/DermFM-Zero'`、notebook 第一段、`linear_probe/linear_eval.py` 中的 `models=('open_clip_hf-hub:redlessone/DermFM-Zero')`）都直接从该 HF 私仓拉权重。**没有 HF token，整套复现脚本一行都跑不起来**——无论 zero-shot、linear-probing 还是 example notebook 都会在 `open_clip.create_model_and_transforms('hf-hub:redlessone/DermFM-Zero')` 处 401。

> 复现需要：作者另行通过邮件/审稿渠道发放一个对 `redlessone/DermFM-Zero` 仓库有 `repo.content.read` 权限的 fine-grained HF token，并在 README 中说明 `huggingface-cli login` 步骤。当前 README 完全没提这一步。

本次本人使用作者提供的 fine-grained token（`accessToken.displayName=DermFM-Zero-NBME-review`，scoped 到 `redlessone/DermFM-Zero`）验证通过。

---

## 1. `conda create -n dermfm-zero python=3.9.20` 成功，无问题。

---

## 2. PyTorch 版本与 Blackwell GPU 不兼容

`requirements.txt` 第一行：

```
torch>=1.9.0
```

直接 `pip install -r requirements.txt` 拉到的是 PyPI 默认的 torch wheel（一般是 cu12.x，但 PyPI 默认 wheel **不含 sm_120/sm_121 编译目标**）。在 GB10 上运行会得到：

```
NVIDIA GB10 with CUDA capability sm_121 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

**绕过方法**：在 `pip install -r requirements.txt` 之前先单独装 cu128 wheel：

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

实测装到 `torch-2.7.1+cu128 / torchvision-0.22.1`，可以正常用 GB10。

> 建议作者在 README/requirements 里固定 `torch==2.7.1+cu128`（或至少 `torch>=2.7,<2.8`），并提示 Blackwell 用户用 `--index-url https://download.pytorch.org/whl/cu128`。否则非作者本机的硬件几乎一定踩坑。

---

(后续步骤继续追加)
