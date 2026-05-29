# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

本文件为 Claude Code 在此仓库中工作时提供指引。

## Setup

```bash
uv venv --python=3.10 && source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
export STABLEWM_HOME=/path/to/storage  # defaults to ~/.stable-wm/
```

## Commands

```bash
# Train (data= selects config/train/data/*.yaml)
python train.py data=pusht
python train.py data=cube
python train.py data=tworoom
python train.py data=dmc

# Evaluate (policy= is relative to $STABLEWM_HOME, no _object.ckpt suffix)
python eval.py --config-name=pusht.yaml policy=pusht/lewm
python eval.py --config-name=cube.yaml policy=cube/lewm
```

WandB must be configured in `config/train/lewm.yaml` (`wandb.config.entity` and `wandb.config.project`) before training.

训练前需在 `config/train/lewm.yaml` 中配置 WandB（`wandb.config.entity` 和 `wandb.config.project`）。

## Architecture

The model (`jepa.py:JEPA`) has four components wired together:
- **Encoder**: ViT-Tiny (no pretrained weights) → CLS token (192-dim)
- **Projector**: `MLP` with `BatchNorm1d` — maps encoder output to latent space
- **ARPredictor**: causal Transformer (depth=6, heads=16) conditioned on action embeddings via **AdaLN-zero** (`module.ConditionalBlock`); predicts next latent from a rolling window of `history_size=3` past embeddings
- **Pred projector**: another `MLP` with `BatchNorm1d` applied to predicted embeddings

Action encoding: `module.Embedder` uses Conv1d + MLP to embed action sequences, then passed as the condition `c` in `ARPredictor.forward(x, c)`.

模型（`jepa.py:JEPA`）由四个组件串联构成：
- **Encoder**：ViT-Tiny（不使用预训练权重）→ CLS token（192维）
- **Projector**：带 `BatchNorm1d` 的 `MLP`，将编码器输出映射到隐空间
- **ARPredictor**：因果 Transformer（depth=6，heads=16），通过 **AdaLN-zero**（`module.ConditionalBlock`）以 action 嵌入为条件；基于最近 `history_size=3` 步的嵌入滚动窗口预测下一时刻的隐变量
- **Pred projector**：另一个带 `BatchNorm1d` 的 `MLP`，作用于预测嵌入

Action 编码：`module.Embedder` 用 Conv1d + MLP 将动作序列编码，作为条件 `c` 传入 `ARPredictor.forward(x, c)`。

## Training Objective

Two losses only (`train.py:lejepa_forward`):

```python
pred_loss  = (pred_emb - tgt_emb).pow(2).mean()          # next-step MSE
sigreg_loss = SIGReg()(emb.transpose(0, 1))               # Gaussian regularizer
loss = pred_loss + 0.09 * sigreg_loss
```

`SIGReg` (`module.py`) is a Sketch Isotropic Gaussian Regularizer: it projects embeddings onto random unit vectors, then measures deviation from a standard Gaussian characteristic function via the Epps-Pulley statistic. This single regularizer replaces EMA, contrastive losses, and stop-gradients used in other JEPAs.

仅两项损失（`train.py:lejepa_forward`）：

`SIGReg`（`module.py`）是 Sketch Isotropic Gaussian Regularizer：将嵌入投影到随机单位向量上，通过 Epps-Pulley 统计量衡量与标准高斯特征函数的偏差。这一单项正则器取代了其他 JEPA 中使用的 EMA、对比损失和 stop-gradient 等复杂组件。

## Planning / Inference

`JEPA.get_cost(info_dict, action_candidates)` is the MPC entry point:
1. Encodes goal pixels → `goal_emb`
2. Calls `rollout()` which autoregressively predicts future latents for each action candidate (batch dim `S`)
3. Returns MSE cost between final predicted embedding and goal embedding, shape `(B, S)`

The `history_size` window is truncated to the last 3 steps during rollout (`emb[:, -HS:]`).

`JEPA.get_cost(info_dict, action_candidates)` 是 MPC 的入口：
1. 编码目标图像 → `goal_emb`
2. 调用 `rollout()`，对每个 action 候选（batch 维度 `S`）自回归预测未来隐变量
3. 返回最终预测嵌入与目标嵌入之间的 MSE 代价，形状为 `(B, S)`

rollout 时将历史窗口截断为最近 3 步（`emb[:, -HS:]`）。

## Configuration System

Hydra configs live in `config/train/` and `config/eval/`. Key overridable values:
- `data=<name>` selects a data config
- `history_size`, `num_preds`, `embed_dim` are shared across model and data configs via Hydra interpolation
- `cfg.model.action_encoder.input_dim` is set dynamically from `frameskip × action_dim` at runtime (not in yaml)

Checkpoints are saved to `$STABLEWM_HOME/checkpoints/<job_id>/` as both `_object.ckpt` (full Python object) and `weights_epoch_N.pt` (state dict).

Hydra 配置位于 `config/train/` 和 `config/eval/`。关键可覆盖项：
- `data=<name>` 选择数据配置
- `history_size`、`num_preds`、`embed_dim` 通过 Hydra 插值在模型与数据配置间共享
- `cfg.model.action_encoder.input_dim` 在运行时由 `frameskip × action_dim` 动态设置（不写在 yaml 中）

checkpoint 保存至 `$STABLEWM_HOME/checkpoints/<job_id>/`，同时存为 `_object.ckpt`（完整 Python 对象）和 `weights_epoch_N.pt`（state dict）。

## External Libraries

- `stable_pretraining` (`spt`): provides `spt.Module`, `spt.Manager`, ViT backbone (`spt.backbone.utils.vit_hf`), data transforms, and the training loop abstraction
- `stable_worldmodel` (`swm`): provides `swm.World`, `swm.policy.*`, `swm.data.*`, MPC solvers (CEM/Adam), and `swm.wm.utils.load_pretrained` / `save_pretrained`

- `stable_pretraining`（`spt`）：提供 `spt.Module`、`spt.Manager`、ViT backbone（`spt.backbone.utils.vit_hf`）、数据变换和训练循环抽象
- `stable_worldmodel`（`swm`）：提供 `swm.World`、`swm.policy.*`、`swm.data.*`、MPC 求解器（CEM/Adam）以及 `swm.wm.utils.load_pretrained` / `save_pretrained`
