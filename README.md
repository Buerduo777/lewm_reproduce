
# LeWorldModel — 复现项目
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

> **本仓库是对原始 [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) 的复现，新增了：**
> - `reproduce.ipynb`：兼容 Mac（MPS）和 Google Colab（T4 GPU）的完整复现 notebook
> - `ANALYSIS.md`：中文技术解析文档（含 Mermaid 架构图、算法推导）
> - `CLAUDE.md`：双语项目指引（面向 Claude Code）
> - 完整的环境配置踩坑记录与解决方案

原论文作者：[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints & Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

---

## 快速复现（reproduce.ipynb）

打开 `reproduce.ipynb`，按以下步骤运行：

| 环境 | MAX_EPOCHS | BATCH_SIZE | 预计时间 |
|:---:|:---:|:---:|:---:|
| Mac 本地验证（MPS） | 1 | 32 | ~5–15 分钟 |
| Google Colab T4 完整训练 | 100 | 128 | ~3–5 小时 |

### 环境安装

```bash
# 推荐使用 uv
uv venv --python=3.10 && source .venv/bin/activate
uv pip install "stable-worldmodel[train,env]"

# 修复已知依赖冲突（numpy 2.x / pyarrow 版本不兼容）
uv pip install "numpy<2" "pyarrow>=12,<14" "datasets>=2.0,<3.0"
```

> **注意（zsh 用户）**：安装时需加引号，否则方括号会被 zsh 当作 glob 展开：
> ```bash
> uv pip install "stable-worldmodel[train,env]"   # ✓
> uv pip install stable-worldmodel[train,env]      # ✗ zsh: no matches found
> ```

### 数据集下载

数据集托管于 HuggingFace，国内用户推荐使用镜像站下载：

```bash
mkdir -p data

# 使用 hf-mirror.com（国内可访问）+ HF token
wget -c "https://hf-mirror.com/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst" \
  -O ./data/pusht_expert_train.h5.zst \
  --header="Authorization: Bearer YOUR_HF_TOKEN" \
  --progress=bar:force

# 解压（需要安装 zstd：brew install zstd）
zstd -d ./data/pusht_expert_train.h5.zst -o ./data/pusht_expert_train.h5 --progress
```

> HuggingFace token 在 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 免费生成。

---

## 原项目使用说明

### 安装

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install "stable-worldmodel[train,env]"
```

### 数据

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

### 训练

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

```bash
python train.py data=pusht
```

### 评估

```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

---

## 预训练 Checkpoint

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

---

## 引用

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```
